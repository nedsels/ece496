#include <common/task.h>
#include <benchmarks/benchmarks.cuh>

template <typename GraphType>
struct BFS {
	using index_type = typename GraphType::index_type;
	using TaskType = Task<index_type>;

    // Initialize; must be called before any other functions
    // Alloc memory on host and device, and reset distances to
    // initial values. Should be called before copy_to_device().
	Benchmarks::Status init(GraphType& graph);

	// Copy distances from host to device after init().
	Benchmarks::Status copy_to_device();

	// Copy distances from host to device.
	Benchmarks::Status copy_to_host();

	// Run BFS on d_graph.
	template <typename SchedulerType>
	Benchmarks::Status solve(
		GraphType d_graph, index_type source,
		SchedulerType* d_scheduler, unsigned grid_size,
		unsigned block_size
	);

	// Print distances by vertex on host
	void print_distances();

	// Dealloc on both host and device.
	Benchmarks::Status dealloc();

	index_type nnodes;
	unsigned distances_bytes = 0;
	unsigned* h_distances = nullptr; /* host */
	unsigned* d_distances = nullptr; /* device */
	bool initialized = false;
};

#include <benchmarks/bfs.cut>
