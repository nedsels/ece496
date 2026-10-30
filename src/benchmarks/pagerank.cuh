#pragma once
#include <common/task.h>
#include <benchmarks/benchmarks.cuh>

// Implements PageRank delta (aka residual)
// TODO: Maybe experiment with traditional PageRank as well
// Typical values for parameters:
// damping factor = 0.85
// epsilon = 0.0001
template <typename GraphType>
struct PAGERANK {
    using index_type = typename GraphType::index_type;
    using weight_type = typename GraphType::edge_data_type;
    // TODO: Priority value unused
    using TaskType = Task<index_type, unsigned>;

    // Initialize; must be called before any other functions
    // Alloc memory on host and device, and reset distances to
    // initial values. Should be called before copy_to_device().
    Benchmarks::Status init(
        const GraphType& graph, const Benchmarks::BenchmarkConfig& config
    );

    // Copy distances from host to device after init().
    Benchmarks::Status copy_to_device();

    // Copy distances from host to device.
    Benchmarks::Status copy_to_host();

    // Run SSSP on d_graph.
    template <typename SchedulerType>
    Benchmarks::Status solve(
        GraphType d_graph, SchedulerType* d_scheduler, 
        unsigned grid_size, unsigned block_size
    );

    // Print distances by vertex on host
    void print_result();

    // Dealloc on both host and device.
    Benchmarks::Status dealloc();

    index_type nnodes;
    unsigned rank_bytes = 0;
    float* h_ranks = nullptr;  /* Host */
    float* d_ranks =  nullptr; /* Device */
    float* h_residuals = nullptr;
    float* d_residuals = nullptr;
    float damping_factor = 0.0;
    float epsilon = 0.0;
    bool initialized = false;
};

#include <benchmarks/pagerank.cut>