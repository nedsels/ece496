#pragma once
#include <common/task.h>
#include <benchmarks/benchmarks.cuh>

// TODO: clean up benchmarks to have common template they inherit off of
// Each will will have roughly same API
// This version only computes path length. It does not reconstruct the path
template <typename GraphType>
struct A_STAR {
    using index_type = typename GraphType::index_type;
    using weight_type = typename GraphType::edge_data_type;
    // f score is a float since h score is a float
    using TaskType = Task<index_type, float>;

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
    index_type src, dst;
    unsigned distances_bytes = 0;
    // Actual distances (g score)
    weight_type* h_distances = nullptr;  /* Host */
    weight_type* d_distances =  nullptr; /* Device */
    bool initialized = false;
};

#include <benchmarks/a_star.cut>