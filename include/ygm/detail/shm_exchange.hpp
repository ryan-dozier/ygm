#ifndef SHM_BUFFER_HPP
#define SHM_BUFFER_HPP

#include <assert.h>
#include <array>
#include <atomic>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <sys/shm.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include <ygm/comm.hpp>
#include <ygm/detail/byte_vector.hpp>
#include <ygm/detail/comm_environment.hpp>
#include <ygm/detail/comm_stats.hpp>
#include <ygm/detail/layout.hpp>

namespace ygm {
namespace shm {
#define MAX_RANKS 256
#define CACHELINE 64
static size_t max_msg_size;

/** 
 *  @brief 
 *  Atomic Counter Struct I need to do some playing with if we can just use alignas during the 
 *  mmap portion of the code as that would remove the need for this. We need each counter to be on
 *  its own cacheline so that processes don't thrash against eachother trying to update counters next
 *  to eachother in the array. The helper functions here are from the underying atomic, just here
 *  for code readability to not have to do array[i].cnt.load()
 */
struct atomic_counters {
public:
  inline size_t load(const std::optional<std::memory_order> o = std::nullopt) const { 
    if(o)
      return cnt.load(o.value());
    else
      return cnt.load(); 
  }
  inline void store(const size_t n, const std::optional<std::memory_order> o = std::nullopt) {
    if(o)
      cnt.store(n, o.value());
    else
      cnt.store(n); 
}
  inline size_t fetch_add(const size_t n, const std::optional<std::memory_order> o = std::nullopt) {
    if(o)
      return cnt.fetch_add(n, o.value());
    else
      return cnt.fetch_add(n); 
  }
private:
  alignas(CACHELINE) std::atomic<size_t> cnt;
};

struct aligned_integer {
public:
  size_t get_value() const { return value; }
  void store_and_synchronize(const size_t n) { value = n; __sync_synchronize(); }
  void add_and_synchronize(const size_t n) { value += n; __sync_synchronize(); }
private:
  alignas(CACHELINE) size_t value = 0;
};

/**
 * @brief 
 * What it aims to do is reduce contention and busy waiting on atomics by having the process wait
 * to execute. If it tries an operation and fails the next time it will wait twice as long up to a
 * specified max_delay
 * @brief Constructs a backoff_helper with a specified maximum delay.
 * @param max The maximum delay for backoff.
 * 
 * @todo: Run some tests with and without, the backoff helper might be outdated with our new model.
 * in the past we ran into more areas of contention and expensive operations when calling to the 
 * filestystem to check if the new shm region was ready
 */ 
struct backoff_helper {
  backoff_helper() : MAX_DELAY(64) { m_delay = 1; }

  backoff_helper(const int max) : MAX_DELAY(max) { m_delay = 1; }
  backoff_helper(backoff_helper&)        = default;
  backoff_helper(const backoff_helper&)  = default;
  backoff_helper(backoff_helper&&)       = default;
  backoff_helper& operator=(const backoff_helper& rhs) = default;

  ~backoff_helper() {}
  void backoff() {
    for (int i = 0; i < m_delay; i++) __asm__("nop\n\t"); // do nothing
    if (m_delay < MAX_DELAY) m_delay <<= 1;
  }
  void reset() { m_delay = 1; }
  int m_delay;
  const int MAX_DELAY;
};


/**
 * @brief SHM buffer for YGM, this structure is designed for many-producer single-consumer. It's 
 * designed as a shared memmory circular buffer for ranks on the same compute node to communicate
 * between eachother. The buffer supports variable msg sizes (insert and read operations are in
 * bytes).
 * @typedef std::byte, just a placeholder for std::byte, allowed for easier unit tests using char
 */
class shm_exchange {
private:
  struct shm_filenames {
    std::string m_data_fname;
    std::string m_reserve_fname;
    std::string m_written_fname;
    std::string m_read_fname;
  };

public:
  shm_exchange(shm_exchange&)        = default;
  shm_exchange(const shm_exchange&)  = default;
  shm_exchange(shm_exchange&&)       = default;
  shm_exchange& operator=(const shm_exchange& rhs) = default;

  shm_exchange(const ygm::detail::layout& layout, const detail::comm_environment& env, detail::comm_stats& stats) : 
                            m_local_rank(layout.local_id()), m_local_size(layout.local_size()), m_max_read_size(env.shm_max_buffer_read), 
                            m_panic(env.buffer_size), m_panic_read_size(env.shm_panic_read_size), m_layout(layout), m_stats(stats) {
    build_shm_exchange(env.shm_buffer_size);
  }

  /* This constructor was going to be used as a standalone without a ygm::comm for validation, however to track how often
   * a local communication skips our shm buffer due to the msg size and uses MPI I'm adding the layout to handle a "can_use_shm()" function

  shm_exchange(const size_t local_id, const size_t local_size, const size_t shm_size, const size_t max_read_size, const size_t panic_size, const size_t panic_read_size) :
                          m_local_rank(local_id), m_local_size(local_size), m_max_read_size(max_read_size), m_panic(panic_size), 
                          m_panic_read_size(panic_read_size) {
    build_shm_exchange(shm_size);
  } */

  /**
   * @brief This private block contains helper functions for the constructor to initialize the shm exchange
   * they should not be called outside of the constructor.
   */
private:
  void build_shm_exchange(size_t global_shm_size) {
    initialize_filenames();
    initialize_page_aligned_sizes(global_shm_size / m_local_size);
    initialize_atomic_counters();
    initialize_shared_memory_region();

    // Ensure each shm region is created and populated by the rank which will be reading from it
    MPI_Barrier(MPI_COMM_WORLD);
    initialize_remote_shared_memory_regions();
    MPI_Barrier(MPI_COMM_WORLD);
  }
  /**
   * @brief Initializes the filenames for shared memory regions and atomic counters.
   */
  void initialize_filenames() {
    m_filenames.m_data_fname = "ygm_shm_exchange_";
    m_filenames.m_reserve_fname = "ygm_shm_reserve";
    m_filenames.m_written_fname = "ygm_shm_tail";
    m_filenames.m_read_fname = "ygm_shm_head";
  }

  /**
   * @brief Calculates and sets the page-aligned sizes for the shared memory buffer and atomic counter arrays.
   * 
   * @param shm_size The size of the shared memory buffer.
   */
  inline void initialize_page_aligned_sizes(const size_t shm_size) {
    auto pagesize = getpagesize();

    // Calculate the page-aligned size for the shared memory buffer
    m_page_aligned_buffer_size = ((shm_size + pagesize - 1) / pagesize) * pagesize;

    // Calculate the page-aligned size for the atomic counter arrays
    auto countersize = sizeof(atomic_counters) * MAX_RANKS;
    m_page_aligned_counter_size = ((countersize + pagesize - 1) / pagesize) * pagesize;
    max_msg_size = m_page_aligned_buffer_size / 2;
    if(m_max_read_size < 0 || m_max_read_size > shm_size) m_max_read_size = m_page_aligned_buffer_size;
  }

  /**
   * @brief Initializes the atomic counters for reserved, written, and read bytes.
   */
  inline void initialize_atomic_counters() {
    shm_unlink(m_filenames.m_reserve_fname.c_str());
    shm_unlink(m_filenames.m_written_fname.c_str());
    shm_unlink(m_filenames.m_read_fname.c_str());
    MPI_Barrier(MPI_COMM_WORLD);
    m_reserved_bytes = open_new_shm_region<atomic_counters>(m_filenames.m_reserve_fname.c_str(), m_page_aligned_counter_size);
    m_reserved_bytes[m_local_rank].store(0);

    m_written_bytes = open_new_shm_region<atomic_counters>(m_filenames.m_written_fname.c_str(), m_page_aligned_counter_size);
    m_written_bytes[m_local_rank].store(0);

    m_read_bytes = open_new_shm_region<aligned_integer>(m_filenames.m_read_fname.c_str(), m_page_aligned_counter_size);
    m_read_bytes[m_local_rank].store_and_synchronize(0);
  }

  /**
   * @brief Initializes the shared memory region for the local rank.
   */
  inline void initialize_shared_memory_region() {
    std::string fname = get_rank_filename((const int) m_local_rank);
    shm_unlink(fname.c_str());
    m_data[m_local_rank] = open_new_shm_region<std::byte>(fname.c_str(), m_page_aligned_buffer_size);
  }

  /**
   * @brief Initializes the shared memory regions for remote ranks.
   */
  inline void initialize_remote_shared_memory_regions() {
    for (int i = 0; i < m_local_size; i++) {
      if (i != m_local_rank) {
        std::string fname = get_rank_filename((const int) i);
        m_data[i] = open_new_shm_region<std::byte>(fname.c_str(), m_page_aligned_buffer_size);
      }
    }
  }

  inline std::string get_rank_filename(const int rank) const {
    return m_filenames.m_data_fname + std::to_string(rank);
  }

public:
  ~shm_exchange() {
    if (m_data[m_local_rank] != nullptr)
      munmap(m_data[m_local_rank], m_page_aligned_buffer_size);

    munmap(m_reserved_bytes, m_page_aligned_counter_size);
    munmap(m_written_bytes, m_page_aligned_counter_size);

    // Ensure all processes reach this point before unlinking shared memory regions
    int finalized;
    MPI_Finalized(&finalized);
    if(!finalized) MPI_Barrier(MPI_COMM_WORLD);

    if (m_local_rank == 0) {
      shm_unlink(m_filenames.m_reserve_fname.c_str());
      shm_unlink(m_filenames.m_written_fname.c_str());
      shm_unlink(m_filenames.m_read_fname.c_str());
    }
    shm_unlink(std::string(get_rank_filename((const int) m_local_rank)).c_str());
    if(!finalized)
      MPI_Barrier(MPI_COMM_WORLD);
  }

  inline size_t size() const { return m_panic.size() + this->shm_size(); }

  inline bool bytes_available() const { return (this->size() > 0) ? true : false; }
   
  /**
   * @brief Returns a ratio of the buffer utilization. This function can return over 1.0 (100%+) if
   * the buffer is full, and the panic buffer has been utilized.
   * 
   * @return double 
   */
  inline double utilized() const { return static_cast<double>(this->size()) / m_page_aligned_buffer_size; }

  /**
   * @brief Returns the maximum size of the shared memory buffer.
   * 
   * @return size_t 
   */
  inline bool can_use_shm(const int dest, const size_t msgsize) {
    bool can_use = m_layout.is_local(dest) && msgsize <= max_msg_size;
    if (!can_use) m_stats.shm_skip();
    return can_use;
  }

  /**
   * @brief Used by the producers. Inserts msgsize bytes into the destination shared buffer. This
   * function  is guarenteed to succeed so no return value.
   * 
   * @param dest destination write
   * @param msg container of outgoing msgs
   * @param msgsize size of the container
   */
  inline void send(const int dest, std::byte* msg, const size_t msgsize) {
    if (msgsize < 0)
      throw std::runtime_error("SHM Buffer: Invalid msgsize detected. Size: " + std::to_string(msgsize));
    if(dest < 0 || dest >= m_local_size)
      throw std::runtime_error("SHM Buffer: Invalid destination detected. Dest: " + std::to_string(dest));
    if (msgsize == 0) return;
    shm_send(dest, (const std::byte*) msg, msgsize);
    m_stats.shm_send(dest, msgsize);
  }

  inline void send(const int dest, std::shared_ptr<ygm::detail::byte_vector>& buffer) {
    send(dest, buffer->data(), (const size_t) buffer->size());
  }

  /**
   * @brief Used by the consuming process. Reads up to buffersize bytes from the shared structure.
   * It returns the number of bytes that were actually read.
   * 
   * @param buffer contiguous storage to read from the shm region
   * @param buffer_size size of the contiguous storage
   * @return size_t bytes actaully read into the buffer
   */
  inline size_t receive(std::shared_ptr<ygm::detail::byte_vector>& buffer) {
    size_t receive_amount = this->size();
    if(receive_amount > 0) {
      shm_receive(receive_amount - m_panic.size());
      buffer->swap(m_panic);
      m_panic.clear();
      YGM_ASSERT_RELEASE(m_panic.size() == 0);
      YGM_ASSERT_RELEASE(buffer->size() == receive_amount);
      m_stats.shm_receive(m_local_rank, receive_amount);
    }
    return receive_amount;
  }

private:
  /**
   * @brief Returns the size of the data in the shared memory buffer.
   * 
   * @return The number of bytes in the shared memory buffer.
   */
  inline size_t shm_size() const {
    const size_t written_bytes = m_written_bytes[m_local_rank].load(std::memory_order_relaxed);
    const size_t read_bytes = m_read_bytes[m_local_rank].get_value();
    return written_bytes - read_bytes;
  }

  /**
   * @brief Writes data to the shared memory (shm) region.
   * 
   * @param dest The destination rank to send a shm msg to.
   * @param msg Pointer to the message data to be written.
   * @param msgsize The size of the message data in bytes.
   */
  void shm_send(const int dest, const std::byte* msg, const size_t msgsize) {
    // Grab the current reserved index and increment by the msgsize
    size_t reserve_start = m_reserved_bytes[dest].fetch_add(msgsize);
    size_t written_bytes = 0;

    do {
      // From the full index grab the buffer id, and the index within the current logical buffer
      size_t cur_index = (reserve_start + written_bytes) % m_page_aligned_buffer_size;

      // in order to handle large msgs we may have to copy in several chunks
      size_t cur_msgsize = msgsize - written_bytes;

      // Check if the current msg will fit within the buffer
      if (cur_index + cur_msgsize > m_page_aligned_buffer_size) {
        cur_msgsize = m_page_aligned_buffer_size - cur_index;
      } 
      
      // Handle potential overlap with the consumer's read position
      handle_consumer_overlap(dest, msg, cur_index, cur_msgsize, written_bytes, reserve_start);

      // copy the bytes that fit into data offset by calculated index
      std::memcpy(m_data[dest] + cur_index, msg + written_bytes, sizeof(std::byte) * cur_msgsize);
      written_bytes += cur_msgsize;

    } while (written_bytes != msgsize);
    __sync_synchronize();

    // Ensure other process make progress before updating the written size
    wait_for_remote_progress(dest, reserve_start);

    // increment the written size, the write becomes visable to other processes here
    m_written_bytes[dest].fetch_add(msgsize);
  }

/**
 * @brief Handles potential overlap with the consumer's read position.
 * When re-using buffers, there is a chance that the write operation might overlap with locations
 * that the consumer has already read. To prevent this, we need to wait until the consumer makes
 * enough progress to allow writing to the buffer.
 * 
 * This function checks if the write operation would cross the consumer's current read position
 * in the circular buffer. For example, the reader could be at index 0 while the writer is at index 64,
 * and it would still be safe to write. This is handled by checking if the consumer's read position
 * falls between the current write index and the index after the partial write.
 *
 * @param dest The destination index in the shared memory region.
 * @param cur_index The current index in the buffer.
 * @param cur_msgsize The current message size.
 * @param written_bytes The number of bytes written so far.
 * @param reserve_start The starting index of the reserved space.
 */
inline void handle_consumer_overlap(const int dest, const std::byte* msg, size_t& cur_index, size_t& cur_msgsize, size_t& written_bytes, const size_t reserve_start) {
  // Check if the consumer's read position falls between the current write index and index of the pending next write.
  // If it does, we need to wait for the consumer to make progress before writing to the buffer.
  // This is done to prevent the writer from overwriting data that the consumer has not yet read.
  // In the mean time, we can copy data from our current read buffer into the panic buffer to not only
  // prevent deadlock, but make some progress while waiting for other processes.
  for (size_t consumed_index = m_read_bytes[dest].get_value() % m_page_aligned_buffer_size;
      (cur_index < consumed_index) && ((cur_index + cur_msgsize) > consumed_index);
       consumed_index = m_read_bytes[dest].get_value() % m_page_aligned_buffer_size) {

    // Calculate the available bytes between the tail and the head
    int cur_avail = consumed_index - cur_index;
    if (cur_avail > 0) {
      // Copy data in the buffer up to the tail
      std::memcpy(m_data[dest] + cur_index, msg + written_bytes, sizeof(std::byte) * cur_avail);
      // Update to reflect the partial write
      written_bytes += cur_avail;
      cur_msgsize -= cur_avail;
      cur_index = (reserve_start + written_bytes) % m_page_aligned_buffer_size;
    } else {
      // Consume to alleviate deadlock if the buffer is more than 50% full
      // TODO: Make a deadlock prone test case to see if this value should be tuneable, an idea for this 
      // could be that each round of iteration we increase the % threshold to do a panic read.
      if (this->utilized() > 0.5) {
          shm_receive(m_panic_read_size);
          m_stats.shm_panic();
      } else {
        m_bh.backoff();
      }
    }
  }
  m_bh.reset();
}

/**
 * @brief Waits for remote messages to make progress.
 * 
 * @param dest The destination index in the shared memory region.
 * @param reserve_start The starting index of the reserved space.
 */
inline void wait_for_remote_progress(int dest, size_t reserve_start) {
  while (m_written_bytes[dest].load() != reserve_start) {
    // Consume to alleviate deadlock if the buffer is more than 50% full
    if (this->utilized() > 0.5) {
      shm_receive(m_panic_read_size);
      m_stats.shm_panic();
    } else {
      m_bh.backoff();
    }
  }
  m_bh.reset();
}

/**
 * @brief Reads from the shared memory (shm) region. This function is responsible for reading data
 *        from the shared memory buffer
 * 
 * @param max_read The maximum number of bytes to read.
 * @return The number of bytes actually read.
 */
  size_t shm_receive(size_t max_read) {
    // Get the current readbytes and writtenbytes pointers.
    // The writtenbytes.load() is our linearization point for reading.
    // The only process which updates the readbytes is the rank owning the buffer.
    size_t cur_tail = m_written_bytes[m_local_rank].load();
    size_t cur_head = m_read_bytes[m_local_rank].get_value();
    size_t read_bytes = 0;

    // Calculate the amount of data available to read
    size_t available_to_read = cur_tail - cur_head;
    if (available_to_read == 0) return 0;
    if (available_to_read > max_read) available_to_read = max_read;

    // Read data from the shared memory buffer
    while (read_bytes < available_to_read) {
      size_t cur_index = (cur_head + read_bytes) % m_page_aligned_buffer_size;
      size_t remaining_bytes = available_to_read - read_bytes;

      // Adjust read size if it extends past the end of the buffer
      if (cur_index + remaining_bytes > m_page_aligned_buffer_size) {
        remaining_bytes = m_page_aligned_buffer_size - cur_index;
      }

      while(remaining_bytes > 0) {
        size_t cur_read = std::min(remaining_bytes, m_max_read_size);
        // copy into the buffer, offset by partial reads, data is offset by the current index
        m_panic.push_bytes(m_data[m_local_rank] + cur_index, sizeof(std::byte) * cur_read);
        read_bytes += cur_read;
        cur_index += cur_read;
        remaining_bytes -= cur_read;
        // Update the partial read, in the non-circular buffer to reduce atomic calls the reader would
        // only update when the whole msg was read. However, other processes may be waiting to write
        // into the region this is currently consuming from.
        m_read_bytes[m_local_rank].add_and_synchronize(cur_read);
      }
    }
    return available_to_read;
  }
 
  /**
   * @brief Opens a shm region with a given filename and size then memory maps onto it. opens with
   * the O_EXL tag. Safe for multiple processes to call on the same filename.
   * 
   * @tparam shm_type 
   * @param filename C-string representing the name of the shared memory region.
   * @param size in bytes of the shared memory region.
   * @return shm_type* 
   */
  template <typename shm_type> shm_type* open_new_shm_region(const char* filename, size_t size) {
    int file = shm_open(filename, O_CREAT | O_RDWR | O_EXCL, 0600);
    if (file == -1 && errno != EEXIST) {
      throw std::runtime_error(std::string("shm_open failed: ") + strerror(errno));
    }
    // if we created the file, set the correct file size
    if (file != -1) {
      if (fallocate(file, 0, 0, size) == -1) {
        close(file);
        throw std::runtime_error(std::string("fallocate failed: ") + strerror(errno));
      }
    } else {
      // if we failed to create the file, open the file for reading
      while (file == -1) {
        file = shm_open(filename, O_RDWR, 0600);
        if (file == -1 && errno != EEXIST) {
          throw std::runtime_error(std::string("shm_open failed: ") + strerror(errno));
        }
      }
    }

    // wait for the file to be the correct size before memory mapping
    struct stat stat_buf;
    do {
      if (fstat(file, &stat_buf) == -1) {
        close(file);
        throw std::runtime_error(std::string("fstat failed: ") + strerror(errno));
      }
      m_bh.backoff();
    } while (stat_buf.st_size != size);
    m_bh.reset();

    // now that the shm_file is the correct size we can memory map to it.
    shm_type* shm_ptr = (shm_type*) mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, file, 0);
    if (shm_ptr == MAP_FAILED) {
      close(file);
      throw std::runtime_error(std::string("mmap failed ") + strerror(errno));
    }

    if (msync(shm_ptr, size, MS_SYNC) == -1) {
      munmap(shm_ptr, size);
      close(file);
      throw std::runtime_error(std::string("msync failed: ") + strerror(errno));
    }
    // yes its safe to close a mapped file
    close(file);
    return shm_ptr;
  } 

  // File sizes, and init info
  size_t                      m_page_aligned_buffer_size;
  size_t                      m_page_aligned_counter_size;

  // File names
  shm_filenames               m_filenames;

  // these need to be shared, consider if renaming these could increase readability
  atomic_counters*            m_reserved_bytes;         // reserves space in shm
  atomic_counters*            m_written_bytes;          // writer location
  aligned_integer*            m_read_bytes;             // reader location
  std::byte*                  m_data[MAX_RANKS];        // shm region for each rank

  // MPI Info
  int                         m_local_rank;
  int                         m_local_size;

  // rank local
  ygm::detail::byte_vector    m_panic;
  size_t                      m_panic_read_size;
  size_t                      m_max_read_size;
  // backoff function, need to run tests with and without it.
  backoff_helper              m_bh;

  detail::comm_stats&         m_stats;
  const ygm::detail::layout&  m_layout;
};  // class shm_exchange
};  // namespace shm
};  // namespace ygm
#endif