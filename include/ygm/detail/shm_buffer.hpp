#ifndef SHM_BUFFER_HPP
#define SHM_BUFFER_HPP
/**
 * @todo Commented out libraries aren't needed to build with ygm as included elsewhere, double check
 * which ones are needed for standalone shm_region (no ygm)
*/
//#include <assert.h>
//#include <array>
#include <atomic>
//#include <cstring>
//#include <deque>
#include <fcntl.h>
//#include <iostream>
//#include <memory>
//#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
//#include <sys/types.h>
#include <unistd.h>
//#include <vector>

#include <ygm/comm.hpp>
#include <ygm/detail/layout.hpp>

namespace shm {
#define MAX_RANKS 256
#define CACHELINE 64

struct recv_buffer {
  std::shared_ptr<std::byte[]> buffer;
  std::shared_ptr<std::byte[]> tmp;
  uint32_t                     cur_size;
  uint32_t                     base_size;

  bool has_resized() {
    return (cur_size != base_size) ? true : false;
  }

  void resize(size_t size) {
    tmp = buffer;
    buffer = std::shared_ptr<std::byte[]>{new std::byte[size]};
    cur_size = size;
  }

  void reset() {
    delete buffer.get();
    buffer = tmp;
    cur_size = base_size;
  }
};

/** 
 *  @brief 
 *  Atomic Counter Struct I need to do some playing with if we can just use alignas during the 
 *  mmap portion of the code as that would remove the need for this. We need each counter to be on
 *  its own cacheline so that processes don't thrash against eachother trying to update counters next
 *  to eachother in the array. The helper functions here are from the underying atomic, just here
 *  for code readability to not have to do array[i].cnt.load()
 */
struct alignas(CACHELINE) atomic_counters {
public:
  inline size_t load() const { return cnt.load(); }
  inline void store(size_t n) { cnt.store(n); }
  inline size_t fetch_add(size_t n) { return cnt.fetch_add(n); }
private:
  std::atomic<size_t> cnt;
};

/**
 * @brief 
 * What it aims to do is reduce contention and busy waiting on atomics by having the process wait
 * to execute. If it tries an operation and fails the next time it will wait twice as long up to a
 * specified max_delay
 * 
 * @todo: Run some tests with and without, the backoff helper might be outdated with our new model.
 * in the past we ran into more areas of contention and expensive operations when calling to the 
 * filestystem to check if the new shm region was ready
 */ 
struct backoff_helper {
  backoff_helper() { m_delay = 1; }
  backoff_helper(int max) : MAX_DELAY(max) { m_delay = 1; }

  ~backoff_helper() {}
  void backoff() {
    for (int i = 0; i < m_delay; i++) __asm__("nop\n\t"); // do nothing
    if (m_delay < MAX_DELAY) m_delay <<= 1;
  }
  void reset() { m_delay = 1; }
  int m_delay;
  const int MAX_DELAY = 64;
};

/**
 * @brief The panic buffer's job is to prevent the shm_buffer from deadlocking. It does this by 
 * allowing blocked insert operations (producers) to consume allowing for at least one process to
 * make progress. It is currently implemented as a deque of memory chunks.
 */
template <typename byte_type> class panic_buffer {
  struct panic_storage {
    std::shared_ptr<std::vector<byte_type>> buffer;
    size_t                                  size;  
  };
public:
  panic_buffer() { m_block_size = 1024; }
  panic_buffer(int panic_size) : m_block_size(panic_size) {}
  ~panic_buffer() {}

  inline size_t size() const { return m_size; }
  inline size_t usage() const { return m_panic_usage; }
  inline size_t get_panic_read_size() const { return m_block_size; }

/**
 * @brief Get the new buffer object.
 * This usage should look like the following:
 * buffer = panic.get_new_buffer();
 * read_bytes = shm_read(buffer, panic.get_panic_read_size());
 * panic.update_size(read_bytes);
 * 
 * 
 * @return byte_type* 
 */
  inline byte_type* get_new_buffer() {
    // maybe unecessary, having a size of 0 is not really supported, so if the size is 0 and a new
    // chunk is requested return the unused chunk at the end
    if(m_storage.size() != 0 && m_storage.back().size == 0) return m_storage.back().buffer->data();
    if (m_reused_chunks.size() == 0) {
      m_storage.push_back(panic_storage{std::make_shared<std::vector<byte_type>>(std::vector<byte_type>(m_block_size)),0});
    } else {
      m_storage.push_back(panic_storage{m_reused_chunks.back(),0});
      m_reused_chunks.pop_back();
    }
    return m_storage.back().buffer->data();
  }

/**
 * @brief updates the size of the last write.
 * This usage should look like the following:
 * buffer = panic.get_new_buffer();
 * read_bytes = shm_read(buffer, panic.get_panic_read_size());
 * panic.update_size(read_bytes);
 * 
 * @param size 
 */
  inline void update_size(size_t size) {
    if(size == 0) return; // i think this edge case is solved on line 115 : if(m_storage.back().size == 0)
    m_storage.back().size = size;
    m_size += size;
    m_panic_usage++;
  }


/**
 * @brief reads up to buffer_size bytes from the panic buffer into the buffer. 
 * 
 * @param buffer pointer to contiguous storage 
 * @param buffer_size size of the contiguous storage
 * @return size_t bytes actaully read into buffer
 */
  size_t read(byte_type* buffer, uint64_t buffer_size) {
    size_t read_amount = 0;
    while (m_size != 0 && read_amount != buffer_size) {
      size_t cur_read = m_storage.front().size;
      std::shared_ptr<std::vector<byte_type>> cur_panic_buffer = m_storage.front().buffer;
      if (read_amount + cur_read <= buffer_size) {
        std::memcpy(buffer + read_amount, cur_panic_buffer->data(), cur_read);
        m_storage.pop_front();
        m_reused_chunks.push_back(cur_panic_buffer);
      } else {
        cur_read = buffer_size - read_amount; // @todo edge cases?
        std::memcpy(buffer + read_amount, cur_panic_buffer->data(), cur_read);
        m_storage.front().size -= cur_read;
        // probably a better way of doing this
        std::memcpy(cur_panic_buffer->data(), cur_panic_buffer->data() + cur_read, m_storage.front().size);
      }
      read_amount += cur_read;
      m_size -= cur_read;
    }
    return read_amount;    
  }

  // debugging
  std::string to_string() {
      std::string out;
      out += "Panic Info:\nGlobal Size: " + std::to_string(m_size) + "\nBlock Size: " + std::to_string(m_block_size) + "\n";
      for(int i = 0; i < m_storage.size(); i++) {
          out += "Buffer: " + std::to_string(i) + "\tSize: " + std::to_string(m_storage[i].size) + "\tData: ";
          std::vector<byte_type>& cur_buff = *(m_storage[i].buffer);
          for(int j = 0; j < m_storage[i].size; j++) {
            out += (char)cur_buff[j];
          }
          out += "\n";
      }
      return out;
  }

private:
  std::deque<panic_storage>                             m_storage;
  std::vector<std::shared_ptr<std::vector<byte_type>>>  m_reused_chunks;
  
  size_t m_size = 0;
  size_t m_block_size = 1024;
  // Stats
  size_t m_panic_usage = 0;
};

/**
 * @brief SHM buffer for YGM, this structure is designed for many-producer single-consumer. It's 
 * designed as a shared memmory circular buffer for ranks on the same compute node to communicate
 * between eachother. The buffer supports variable msg sizes (insert and read operations are in
 * bytes).
 * @typedef byte_type, just a placeholder for std::byte, allowed for easier unit tests using char
 */
template <typename byte_type> class shm_buffer {
public:
  static_assert(sizeof(byte_type) == 1, "shm_buffer requires byte sized type.\n");
  shm_buffer() = delete;

  shm_buffer(const shm_buffer &c) = delete;
  
  shm_buffer(const ygm::detail::layout& layout, const size_t shm_size, const int panic_size) :
                           m_local_rank(layout.local_id()), m_local_size(layout.local_size()), m_panic(panic_size) {
    m_buff_fname = std::string("ygm_shm_buffer_");
    m_res_fname  = std::string("ygm_shm_reserve");
    m_tail_fname = std::string("ygm_shm_tail");
    m_head_fname = std::string("ygm_shm_head");

    auto pagesize = getpagesize();
    // calc the page aligned size for the shm buffer
    auto num_pages = shm_size / pagesize;
    if (shm_size % pagesize != 0) num_pages++;
    m_page_aligned_buffer_size = num_pages * pagesize;

    // calc the page aligned size for the atomic counter arrays
    auto countersize = sizeof(atomic_counters) * MAX_RANKS;
    num_pages = countersize / pagesize;

    if (countersize % pagesize != 0) num_pages++;
    m_page_aligned_counter_size = num_pages * pagesize;

    // create the regions for the atomic counters
    m_reserve = this->open_new_shm_region<atomic_counters>(m_res_fname.c_str(), m_page_aligned_counter_size);
    m_reserve[m_local_rank].store(0);

    m_tail    = this->open_new_shm_region<atomic_counters>(m_tail_fname.c_str(), m_page_aligned_counter_size);
    m_tail[m_local_rank].store(0);

    m_head    = this->open_new_shm_region<atomic_counters>(m_head_fname.c_str(), m_page_aligned_counter_size);
    m_head[m_local_rank].store(0);


    // create the shm regions for the data
    std::string fname = m_buff_fname + std::to_string(m_local_rank);
    m_data[m_local_rank] = this->open_new_shm_region<byte_type>(fname.c_str(), m_page_aligned_buffer_size);
    // technically this barrier is not needed, but it guarentees each shm region is created and
    // populated by the rank which will be reading from it.
    /** @todo mpi subcommunicator */
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < m_local_size; i++) {
      if (i != m_local_rank) {
        fname = m_buff_fname + std::to_string(i);
        m_data[i] = this->open_new_shm_region<byte_type>(fname.c_str(), m_page_aligned_buffer_size);
      }
    }
  }

  ~shm_buffer() {
    if (m_data[m_local_rank] != nullptr)
      munmap(m_data[m_local_rank], m_page_aligned_buffer_size);

    munmap(m_reserve, m_page_aligned_counter_size);
    munmap(m_tail, m_page_aligned_counter_size);
    if (m_local_rank == 0) {
      shm_unlink(m_res_fname.c_str());
      shm_unlink(m_tail_fname.c_str());
      shm_unlink(m_head_fname.c_str());
    }
    shm_unlink(std::string(m_buff_fname + std::to_string(m_local_rank)).c_str());
  }



  inline size_t size() const { return m_panic.size() + this->shm_size(); }

  inline bool bytes_available() const { return (this->size() > 0) ? true : false; }
   
  /**
   * @brief Returns a ratio of the buffer utilization. This function can return over 1.0 (100%+) if
   * the buffer is full, and the panic buffer has been utilized.
   * 
   * @return double 
   */
  inline double utilized() const { return (double)this->size() / m_page_aligned_buffer_size; }

  /**
   * @brief Used by the producers. Inserts msgsize bytes into the destination shared buffer. This
   * function  is guarenteed to succeed so no return value.
   * 
   * @param dest destination write
   * @param msg container of outgoing msgs
   * @param msgsize size of the container
   */
  void insert(int dest, byte_type* msg, size_t msgsize) {
    if (msgsize > 0 && dest < m_local_size) shm_insert(dest, msg, msgsize);
  }

  /**
   * @brief Used by the consuming process. Reads up to buffersize bytes from the shared structure.
   * It returns the number of bytes that were actually read.
   * 
   * @param buffer contiguous storage to read from the shm region
   * @param buffer_size size of the contiguous storage
   * @return size_t bytes actaully read into the buffer
   */
  size_t read(void* buffer, size_t buffer_size) {
    size_t read_amount = 0;
    // check if there is read data in the panic buffer
    if (m_panic.size() != 0) {
      // read any available data in the panic buffer 
      read_amount = m_panic.read((byte_type*)buffer, buffer_size);
    }
    // calculate the remaining space in the buffer
    size_t remaining_buffer = buffer_size - read_amount;
    if (remaining_buffer > 0) {
      // can move this into the next line, it just makes it less readable.
      byte_type* buffer_offset = (byte_type*)buffer + read_amount;
      read_amount += shm_read((void*)buffer_offset, remaining_buffer);
    }
    return read_amount;
  }

// Mostly for debugging purposes, but outputs the current status of the current rank's buffer.
std::string to_string() const {
  size_t head = m_head[m_local_rank].load();
  size_t tail = m_tail[m_local_rank].load();

  std::string result  = std::string("SHM Buffer Info:");
              result += std::string("\nrank:\t" + std::to_string(m_local_rank));
              result += std::string("\nsize:\t" + std::to_string(size()));
              result += std::string("\nutilize:\t" + std::to_string(utilized() * 100) + "%");

              result += std::string("\n\nPanic Buffer Info:");
              result += std::string("\nsize:\t\t" + std::to_string(m_panic.size()));
              result += std::string("\nused:\t" + std::to_string(m_panic.usage()));
  return result;
}


private:
  inline size_t shm_size() const { return m_tail[m_local_rank].load() - m_head[m_local_rank].load(); }

  // Write to the SHM region, see insert(int dest, byte_type* msg, size_t msgsize) above for the
  // full producer process.
  void shm_insert(int dest, byte_type* msg, size_t msgsize) {
    // grab the current reserved index, and increment by the msgsize
    size_t reserve_start = m_reserve[dest].fetch_add(msgsize);
    size_t written_bytes = 0;
    do {
      // from the full index grab the buffer id, and the index within the current logical buffer
      size_t cur_index = (reserve_start + written_bytes) % m_page_aligned_buffer_size;

      // in order to handle large msgs we may have to copy in several chunks
      size_t cur_msgsize = msgsize - written_bytes;

      // check if the current msg will fit within the buffer
      if (cur_index + cur_msgsize > m_page_aligned_buffer_size) {
        cur_msgsize = m_page_aligned_buffer_size - cur_index;
      } 
      
      // when re-using buffers we have the chance to write over locations where the consuming rank
      // has read to so far. for the moment we'll have to wait until the consumer makes enough
      // progress to write to the buffer. we'll look into other methods to potentially improve this
      // here we care the a write will cross where the consumer is currently working because it's a
      // circular buffer the reader could be at index 0 and the writer could be at index 64 and it's
      // safe to write. I believe this solves this by checking if consumed counter falls between the
      // current written index, and the index + the partial write size.
      for (size_t consumed_index = m_head[dest].load() % m_page_aligned_buffer_size;
          (cur_index < consumed_index) && ((cur_index + cur_msgsize) > consumed_index); 
           consumed_index = m_head[dest].load() % m_page_aligned_buffer_size) {

        m_bh.backoff();
        // calc the available bytes between the tail and the head
        int cur_avail = consumed_index - cur_index; // TODO: work out if there are edge cases here
        if (cur_avail > 0) { 
          // copy data in the buffer up to the tail
          std::memcpy(m_data[dest] + cur_index, msg + written_bytes, sizeof(byte_type) * cur_avail);
          // update to reflect the partial write
          written_bytes += cur_avail;
          cur_msgsize -= cur_avail;
          cur_index = (reserve_start + written_bytes) % m_page_aligned_buffer_size;
        } else { // because we're unable to write (produce) we should consume to alleviate deadlock
          int write_amount = m_panic.get_panic_read_size();
          if (write_amount != 0) {
            m_panic.update_size(shm_read((void*)m_panic.get_new_buffer(), write_amount));
          }
        }
      }
      m_bh.reset(); 

      // copy the bytes that fit into data offset by calculated index
      std::memcpy(m_data[dest] + cur_index, msg + written_bytes, sizeof(byte_type) * cur_msgsize);
      written_bytes += cur_msgsize;

    } while (written_bytes != msgsize);
    __sync_synchronize();

    // currently the only process that can make progress is the next sequential msg
    while (m_tail[dest].load() != reserve_start) { 
      // wait for backoff and loads to execute potentially clearing up the problem
      m_bh.backoff();
      int write_amount = m_panic.get_panic_read_size();
      // when serializing only panic if our buffer is 50% full
      if (write_amount != 0 && this->utilized() > 0.5) { 
        m_panic.update_size(shm_read((void*)m_panic.get_new_buffer(), write_amount));
      }
    }
    m_bh.reset();
  
    // increment the written size, the write becomes visable to other processes here
    m_tail[dest].fetch_add(msgsize);
  }

  
  /**
   * @deprecated runs the panic buffer, this section is still under development
   * this was probably really overengineered
   * the idea was to pass the iteration id to the function to try and allow stuck processes to work
   * itself out
   * @param iteration 
   */
  void test_panic_buffer(int iteration) {
    // probably pull these out and define. probably cleaner way to handle this. 
    const int inner_test = 63;
    const int outer_test = 10000;
    size_t self_head = m_head[m_local_rank].load();
    size_t self_tail = m_tail[m_local_rank].load();
    int write_amount;

    if (iteration % inner_test == 0) {
      // check if the available capacity is less than a local buffer size. this should reduce 
      // unecessary memcpys, we only want to use the panic buffer if theres a chance of a stall 
      size_t cur_avail = m_page_aligned_buffer_size - (self_tail - self_head);
      if (cur_avail < m_panic.trigger()) {
        write_amount = m_panic.get_panic_read_size();
        if (write_amount != 0) {
          m_panic.update_size(shm_read((void*)m_panic.get_new_buffer(), write_amount));
        }
      }
    } else if (iteration % outer_test == 0) {
      write_amount = m_panic.get_panic_read_size();
      if (write_amount != 0) {
        m_panic.update_size(shm_read((void*)m_panic.get_new_buffer(), write_amount));
      }
    } 
  }

  /**
   * @brief reads from the shm region, see read_bytes(void* buffer, size_t buffer_size) above for
   * the public function interface
   * 
   * @param buffer 
   * @param buffer_size 
   * @return number of read bytes 
   */
  size_t shm_read(void* buffer, size_t buffer_size) {
    // grab the current head and tail. The tail.load() is our linearization point for reading. 
    // the only process which updates the head is the rank owning the buffer.
    size_t cur_tail = m_tail[m_local_rank].load();
    size_t cur_head = m_head[m_local_rank].load();
    size_t read_size = 0;
    size_t read_bytes = 0;
    // do buffersize check
    size_t read_amount = cur_tail - cur_head;
    if (read_amount == 0) return 0;
    if (read_amount > buffer_size) read_amount = buffer_size;
    while (read_bytes != read_amount) {
      //calculate current buffer, and index within
      size_t cur_index = (cur_head + read_bytes) % m_page_aligned_buffer_size;

      read_size = read_amount - read_bytes;
      // check if the current read will extend past the end of the current buffer
      if (cur_index + read_size > m_page_aligned_buffer_size) {
        // modify the current read size to the end of this buffer
        read_size = m_page_aligned_buffer_size - cur_index;
      }
      // copy into the buffer, offet by partial reads, data is offset by the current index
      std::memcpy((byte_type*)buffer + read_bytes, m_data[m_local_rank] + cur_index, sizeof(byte_type) * read_size);
      read_bytes += read_size;
      // update the partial read, in the non-circular buffer to reduce atomic calls the reader would
      // only update when the whole msg was read. However, other processes may be waiting to write
      // into the region this is currenly consuming from.
      m_head[m_local_rank].fetch_add(read_size);
    }
    return read_amount;
  }
 
  /**
   * @brief Opens a shm region with a given filename and size then memory maps onto it. opens with
   * the O_EXL tag. Safe for mutliple processes to call on the same filename.
   * 
   * @tparam shm_type 
   * @param filename 
   * @param size 
   * @return shm_type* 
   */
  template <typename shm_type> shm_type* open_new_shm_region(const char* filename, size_t size) {
    int file = shm_open(filename, O_CREAT | O_RDWR | O_EXCL, 0600);
    // if we created the file, set the correct file size
    if (file != -1) fallocate(file, 0, 0, size);
    // if we failed to create the file, open the file for reading
    while (file == -1) {
      file = shm_open(filename, O_RDWR, 0600);
    }

    // wait for the file to be the correct size before memory mapping
    struct stat stat_buf;
    do {
      fstat(file, &stat_buf);
      m_bh.backoff();
    } while (stat_buf.st_size != size);
    m_bh.reset();

    // now that the shm_file is the correct size we can memory map to it.
    shm_type* shm_ptr = (shm_type*) mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, file, 0);
    if (shm_ptr == MAP_FAILED) {
      std::cerr << "reserve mmap failed" << std::endl;
      exit(-1);
    }
    int msync_ret = msync(shm_ptr, size, MS_SYNC);
    if (msync_ret != 0) {
      std::cerr << "msync failed" << std::endl;
    }
    // yes its safe to close a mapped file
    close(file);
    return shm_ptr;
  } 

  // File sizes, and init info
  size_t                      m_page_aligned_buffer_size;
  size_t                      m_page_aligned_counter_size;

  //filenames
  std::string                 m_buff_fname;
  std::string                 m_res_fname;
  std::string                 m_tail_fname;
  std::string                 m_head_fname;

  // these need to be shared, consider if renaming these could increase readability
  atomic_counters*            m_reserve;                // reserves space in shm
  atomic_counters*            m_tail;                   // writer location
  atomic_counters*            m_head;                   // reader location
  byte_type*                  m_data[MAX_RANKS];        // shm region for each rank

  // MPI Info
  const int                   m_local_rank;
  const int                   m_local_size;

  // rank local
  panic_buffer<byte_type>     m_panic;

  // backoff function, need to run tests with and without it.
  backoff_helper              m_bh;
};
};
#endif