#include <benchmarks/bfs.cuh>
#include <benchmarks/sssp.cuh>
#include <benchmarks/a_star.cuh>
#include <benchmarks/pagerank.cuh>
#include <graph/csr_graph.cuh>
#include <scheduler/scheduler.cuh>
#include <benchmarks/benchmarks.cuh>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <optional>

namespace {

// Optional flags given after the positional arguments.
struct RunOptions {
  const char* out_path = nullptr; // --out FILE: write distances as raw uint32
  bool print_time = false;        // --time: print timing as one JSON line
  bool verbose = false;           // --verbose: print distances to stdout
};

template <template <typename> class BenchmarkTemplate, typename WeightType, typename NodeDataType>
Benchmarks::Status run_benchmark(
  const char* graph_path, const Benchmarks::BenchmarkConfig& config, 
  SchedulerType scheduler_choice, const cudaDeviceProp& prop,
  const RunOptions& options
);

template <template <typename> class BenchmarkTemplate, typename NodeDataType = std::int32_t>
int run_with_weight(
  const char* graph_path, const char* weight_name, SchedulerType scheduler_choice, 
  const cudaDeviceProp& prop, const Benchmarks::BenchmarkConfig& config, const RunOptions& options
); 

template <typename GraphType>
bool is_valid_config(const Benchmarks::BenchmarkConfig &config, const GraphType& graph);

} // End anonymous namespace

int main(int argc, char* argv[]) {
  // NOTE: Put benchmark specific configurations last so first arguments are consistent for all benchmarks
  // This doesn't check if the user provides the correct number of arguments for a given benchmark
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " <bfs/sssp> <graph.gr> <weight-type> <scheduler> <source>\n"
              << " <a_star> <graph.gr> <weight-type> <scheduler> <source> <destination>\n"
              << " <pr> <graph.gr> <weight-type> <scheduler> <damping factor> <epsilon>\n"
              << " optional flags: [--out FILE] [--time] [--verbose]\n"
              << " weight-type: int32, uint32\n"
              << "scheduler: initial_scheduler\n"
              << "--out FILE: write distances to FILE as raw uint32 values\n"
              << "--time: print {\"kernel_ms\": ..., \"load_ms\": ...}\n"
              << "--verbose: also print distances when --out is given\n";
    return 1;
  }     
  
  RunOptions options;
  for (int i = 6; i < argc; i++) {
    if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
      options.out_path = argv[++i];
    } else if (std::strcmp(argv[i], "--time") == 0) {
      options.print_time = true;
    } else if (std::strcmp(argv[i], "--verbose") == 0) {
      options.verbose = true;
    }
  }

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  try {
    const SchedulerType scheduler_choice = from_string(argv[4]);

    if (std::strcmp(argv[1], "bfs") == 0) {
      const Benchmarks::BenchmarkConfig config = {.source=std::strtol(argv[5], nullptr, 10)};
      return run_with_weight<BFS>(argv[2], argv[3], scheduler_choice, prop, config, options);
    }
    if (std::strcmp(argv[1], "sssp") == 0) {
      const Benchmarks::BenchmarkConfig config = {.source=std::strtol(argv[5], nullptr, 10)};
      return run_with_weight<SSSP>(argv[2], argv[3], scheduler_choice, prop, config, options);
    }
    if (std::strcmp(argv[1], "a_star") == 0) {
      const Benchmarks::BenchmarkConfig config = {.source=std::strtol(argv[5], nullptr, 10), .destination=std::strtol(argv[6], nullptr, 10)};
      return run_with_weight<A_STAR, std::pair<std::int32_t, std::int32_t>>(argv[2], argv[3], scheduler_choice, prop, config, options);
    }
    if (std::strcmp(argv[1], "pr") == 0) {
      const Benchmarks::BenchmarkConfig config = {.source=std::nullopt, .destination=std::nullopt, .damping_factor=std::stof(argv[5]), .epsilon=std::stof(argv[6])};
      return run_with_weight<PAGERANK>(argv[2], argv[3], scheduler_choice, prop, config, options);
    }

    std::cerr << "Unsupported benchmark: " << argv[1] << '\n';
    return 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

namespace {

template <typename GraphType>
bool is_valid_config(const Benchmarks::BenchmarkConfig &config, const GraphType& graph) {
  if (config.source) {
    const auto source = config.source.value();
    if (source < 0 || source >= graph.nnodes) {
      std::cerr << "Source vertex is outside the graph.\n";
      return false;
    }
  }
  if (config.destination) {
    const auto dest = config.destination.value();
    if (dest < 0 || dest >= graph.nnodes) {
      std::cerr << "Destination vertex is outside the graph.\n";
      return false;
    }
  }
  return true;
}

// TODO: create Benchmark std::variants. Today we have scheduler variants, but
// what we want is for the benchmark to be chosen at runtime but the scheduler
// to be chosen at compile time.
template <template <typename> class BenchmarkTemplate, typename WeightType, typename NodeDataType>
Benchmarks::Status run_benchmark(
  const char* graph_path, const Benchmarks::BenchmarkConfig& config,
  SchedulerType scheduler_choice, const cudaDeviceProp& prop, const RunOptions& options
) {
  using GraphType = CSRGraph<WeightType, NodeDataType>;
  using Benchmark = BenchmarkTemplate<GraphType>;
  using TaskType = typename Benchmark::TaskType;

  // Load the graph into host memory and create its device copy.
  const auto load_start = std::chrono::steady_clock::now();
  GraphType h_graph, d_graph;
  h_graph.init();
  if (h_graph.readFromGR(graph_path) != 0) {
    std::cerr << "Unable to load graph.\n";
    return Benchmarks::Status::ERROR;
  }

  const double load_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - load_start)
                            .count();

  if (!is_valid_config<GraphType>(config, h_graph)) {
    h_graph.dealloc();
    return Benchmarks::Status::ERROR;
  }

  if (!h_graph.copy_to_gpu(d_graph)) {
    std::cerr << "Copying graph to GPU failed.\n";
    h_graph.dealloc();
    return Benchmarks::Status::ERROR;
  }

  // Allocate and initialize the benchmark's host/device working buffers.
  Benchmark benchmark;
  if (benchmark.init(h_graph, config) == Benchmarks::Status::ERROR || 
      benchmark.copy_to_device() == Benchmarks::Status::ERROR
  ) {
    benchmark.dealloc();
    d_graph.dealloc();
    h_graph.dealloc();
    return Benchmarks::Status::ERROR;
  }

  Benchmarks::Status result = Benchmarks::Status::SUCCESS;
  try {
    Scheduler<TaskType> h_scheduler = get_scheduler<TaskType>(scheduler_choice);

    // HYPERPARAMETER
    // Reserve 1/8 of free VRAM (completely arbitrary)
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const unsigned scheduler_byte_capacity =
        static_cast<unsigned>(free_bytes / 8);

    // HYPERPARAMETER
    // Use one block per half of SMs and half the maximum number of warps per
    // block (also arbitrary)
    const unsigned num_blocks = prop.multiProcessorCount / 2;
    const unsigned warps_per_block =
        prop.maxThreadsPerBlock / prop.warpSize / 2;
    const unsigned num_warps = warps_per_block * num_blocks;

    std::visit(
        [&](auto& scheduler_impl) {
          scheduler_impl.init(scheduler_byte_capacity, num_blocks, num_warps);
          auto* d_scheduler = scheduler_impl.device_setup(prop);
          if (d_scheduler == nullptr ||
              benchmark.solve(d_graph, d_scheduler, num_blocks,
                              warps_per_block * prop.warpSize) == Benchmarks::Status::ERROR ||
              benchmark.copy_to_host() == Benchmarks::Status::ERROR) {
            result = Benchmarks::Status::ERROR;
          }
          scheduler_impl.device_dealloc();
        },
        h_scheduler);

    if (result == Benchmarks::Status::SUCCESS) {
      if (options.out_path != nullptr &&
          benchmark.write_results(options.out_path) == Benchmarks::Status::ERROR) {
        result = Benchmarks::Status::ERROR;
      }
      // Without --out, printing is the only way to see the result.
      if (options.verbose || options.out_path == nullptr) {
        benchmark.print_result();
      }
      // Printed last so callers can parse the final stdout line as JSON.
      if (options.print_time) {
        std::cout << "{\"kernel_ms\": " << benchmark.kernel_ms
                  << ", \"load_ms\": " << load_ms << "}" << std::endl;
      }
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    result = Benchmarks::Status::ERROR;
  }

  benchmark.dealloc();
  d_graph.dealloc();
  h_graph.dealloc();
  return result;
}

// NOTE: NodeDataType is currently only needed for A* for coordinates used to compute h score
// Default this to int for now. When building the graph, the node data is optional and only done if
// specified in the .gr file
template <template <typename> class BenchmarkTemplate, typename NodeDataType>
int run_with_weight(
  const char* graph_path,
  const char* weight_name,
  SchedulerType scheduler_choice,
  const cudaDeviceProp& prop,
  const Benchmarks::BenchmarkConfig& config,
  const RunOptions& options
) {
  if (std::strcmp(weight_name, "int32") == 0) {
    return static_cast<int>(
      run_benchmark<BenchmarkTemplate, std::int32_t, NodeDataType>(graph_path, config, scheduler_choice, prop, options)
    );
  }
  if (std::strcmp(weight_name, "uint32") == 0) {
    return static_cast<int>(
      run_benchmark<BenchmarkTemplate, std::uint32_t, NodeDataType>(graph_path, config, scheduler_choice, prop, options)
    );
  }
  // TODO: float weights?
  std::cerr << "Unsupported weight type: " << weight_name << '\n';
  return 1;
}

} // End anonymous namespace
