#include <algorithm>
#include <random>
#include <ygm/collective.hpp>
#include <ygm/comm.hpp>
#include <ygm/container/bag.hpp>
#include <ygm/container/disjoint_set.hpp>
#include <ygm/io/ndjson_parser.hpp>
#include <ygm/utility.hpp>

//
std::vector<std::string> json_filenames{
    "/p/lustre3/llamag/reddit_2023/comments_orig"};

template <typename T>
void pivot_sort(std::vector<T> &in_vec, ygm::comm world) {
  const size_t   samples_per_pivot = 20;
  std::vector<T> to_sort;
  to_sort.reserve(in_vec.size() * 1.1f);

  //
  //  Choose pivots, uses index as 3rd sorting argument to solve issue with lots
  //  of duplicate items
  std::vector<std::pair<T, size_t>> samples;
  std::vector<std::pair<T, size_t>> pivots;
  static auto                      &s_samples = samples;
  static auto                      &s_to_sort = to_sort;
  samples.reserve(world.size() * samples_per_pivot);

  //
  std::default_random_engine rng;

  size_t my_prefix   = ygm::prefix_sum(in_vec.size(), world);
  size_t global_size = ygm::sum(in_vec.size(), world);
  std::uniform_int_distribution<size_t> uintdist{0, global_size - 1};

  for (size_t i = 0; i < samples_per_pivot * (world.size() - 1); ++i) {
    size_t index = uintdist(rng);
    if (index >= my_prefix && index < my_prefix + in_vec.size()) {
      world.async_bcast(
          [](const std::pair<T, size_t> &sample) {
            s_samples.push_back(sample);
          },
          std::make_pair(in_vec[index - my_prefix], index));
    }
  }
  world.barrier();

  ASSERT_RELEASE(samples.size() == samples_per_pivot * (world.size() - 1));
  std::sort(samples.begin(), samples.end());
  for (size_t i = samples_per_pivot - 1; i < samples.size();
       i += samples_per_pivot) {
    pivots.push_back(samples[i]);
  }
  samples.clear();
  samples.shrink_to_fit();

  ASSERT_RELEASE(pivots.size() == world.size() - 1);

  //
  // Partition using pivots
  for (size_t i = 0; i < in_vec.size(); ++i) {
    auto   itr   = std::lower_bound(pivots.begin(), pivots.end(),
                                    std::make_pair(in_vec[i], my_prefix + i));
    size_t owner = std::distance(pivots.begin(), itr);

    world.async(
        owner, [](const T &val) { s_to_sort.push_back(val); }, in_vec[i]);
  }
  world.barrier();

  if (not to_sort.empty()) {
    std::sort(to_sort.begin(), to_sort.end());
  }

  // world.cout("to_sort.size() = ", to_sort.size());

  // //
  // // verify
  // world.barrier();
  // ASSERT_RELEASE(ygm::sum(in_vec.size(), world) ==
  //                ygm::sum(to_sort.size(), world));
  // if (world.rank() < world.size() - 1 && not to_sort.empty()) {
  //   world.async(
  //       world.rank() + 1,
  //       [](const T &val) {
  //         if (not s_to_sort.empty()) {
  //           ASSERT_RELEASE(val <= s_to_sort[0]);
  //         }
  //       },
  //       to_sort.back());
  // }

  world.barrier();
  in_vec.swap(to_sort);
}

int main(int argc, char **argv) {
  ygm::comm world(&argc, &argv);
  {
    world.welcome();
    world.cout0("");

    //
    // Parses directory full of line deliminated JSON in parallel
    ygm::io::ndjson_parser jsonp(world, json_filenames, false, true);

    //
    // strings
    std::vector<std::string> local_strings;

    //
    // Processes each JSON object independently and in parallel
    jsonp.for_all([&local_strings](auto &json_line) {
      if (json_line.count("body")) {
        std::string body = json_line["body"].as_string().c_str();
        local_strings.push_back(body);
      }
    });
    world.barrier();

    world.cout0("Finished reading, ", ygm::sum(local_strings.size(), world));

    ygm::timer timer;
    {
      pivot_sort(local_strings, world);
      world.barrier();
    }
    world.cout0("Sort time = ", timer.elapsed());
    world.barrier();
  }
  return 0;
}

