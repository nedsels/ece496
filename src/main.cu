#include <benchmarks/bfs.cuh>
#include <graph/csr_graph.cuh>
#include <scheduler/scheduler.cuh>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>

// TODO create Benchmark std::variants
template <template <typename> class BenchmarkTemplate, typename WeightType>
int run_benchmark(const char* graph_path, const char* source_text,
                  SchedulerType scheduler_choice, const cudaDeviceProp& prop) {
  using GraphType = CSRGraph<WeightType>;
  using Benchmark = BenchmarkTemplate<GraphType>;
  using TaskType = typename Benchmark::TaskType;
  using Vertex = typename GraphType::index_type;

  // Load the graph into host memory and create its device copy.
  GraphType h_graph, d_graph;
  h_graph.init();
  if (h_graph.readFromGR(graph_path) != 0) {
    std::cerr << "Unable to load graph.\n";
    return 1;
  }

  const long source_value = std::strtol(source_text, nullptr, 10);
  if (source_value < 0 || source_value >= h_graph.nnodes) {
    std::cerr << "Source vertex is outside the graph.\n";
    h_graph.dealloc();
    return 1;
  }
  const Vertex source = static_cast<Vertex>(source_value);
  if (!h_graph.copy_to_gpu(d_graph)) {
    std::cerr << "Copying graph to GPU failed.\n";
    h_graph.dealloc();
    return 1;
  }

  // Allocate and initialize the benchmark's host/device working buffers.
  Benchmark benchmark;
  benchmark.init(h_graph);
  if (!benchmark.setup() || !benchmark.copy_to_device()) {
    benchmark.dealloc();
    d_graph.dealloc();
    h_graph.dealloc();
    return 1;
  }

  int result = 0;
  try {
    Scheduler<TaskType> h_scheduler = get_scheduler<TaskType>(scheduler_choice);

    // HYPERPARAMETER
    // Reserve half of free VRAM (arbitrary, should be chosen more
    // intelligently)
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const unsigned scheduler_byte_capacity =
        static_cast<unsigned>(free_bytes / 2);

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
              !benchmark.solve(d_graph, source, d_scheduler, num_blocks,
                               warps_per_block * prop.warpSize) ||
              !benchmark.copy_to_host()) {
            result = 1;
          }
          scheduler_impl.device_dealloc();
        },
        h_scheduler);

    benchmark.print_distances();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    result = 1;
  }

  benchmark.dealloc();
  d_graph.dealloc();
  h_graph.dealloc();
  return result;
}

int main(int argc, char* argv[]) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " <benchmark> <graph.gr> <source> <weight-type> <scheduler>\n"
              << "benchmark: bfs\n"
              << "weight-type: int32, uint32, float32\n";
    return 1;
  }

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  try {
    if (std::strcmp(argv[1], "bfs") != 0) {
      std::cerr << "Unsupported benchmark: " << argv[1] << '\n';
      return 1;
    }

    const SchedulerType scheduler_choice = from_string(argv[5]);
    if (std::strcmp(argv[4], "int32") == 0) {
      return run_benchmark<BFS, std::int32_t>(argv[2], argv[3],
                                              scheduler_choice, prop);
    }
    if (std::strcmp(argv[4], "uint32") == 0) {
      return run_benchmark<BFS, std::uint32_t>(argv[2], argv[3],
                                               scheduler_choice, prop);
    }
    if (std::strcmp(argv[4], "float32") == 0) {
      return run_benchmark<BFS, float>(argv[2], argv[3], scheduler_choice,
                                       prop);
    }
    std::cerr << "Unsupported weight type: " << argv[4] << '\n';
    return 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
