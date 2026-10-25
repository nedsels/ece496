#include <common/task.h>

template <typename GraphType>
struct BFS {
  using index_type = typename GraphType::index_type;
  using TaskType = Task<index_type>;

  // Initialize; must be called before any other functions
  unsigned init(GraphType& graph);

  // Alloc memory on host and device, and reset distances to
  // initial values. Should be called before
  // copy_to_device().
  unsigned setup();

  // Copy distances from host to device after setup().
  unsigned copy_to_device();

  // Copy distances from host to device.
  unsigned copy_to_host();

  // Run BFS on d_graph.
  template <typename SchedulerType>
  unsigned solve(GraphType d_graph, index_type source,
                 SchedulerType* d_scheduler, unsigned grid_size,
                 unsigned block_size);

  // Print distances by vertex on host
  unsigned print_distances();

  // Dealloc on both host and device.
  unsigned dealloc();

  index_type nnodes;
  unsigned distances_bytes;
  unsigned* h_distances;
  unsigned* d_distances;
};

#include <benchmarks/bfs.cut>
