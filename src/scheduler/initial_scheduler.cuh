#pragma once

#include <graph/csr_graph.cuh>
#include <scheduler/enum.cuh>

// HYPERPARAMTERS
#define NUM_SECTIONS 32

// This scheduler has _num_sections sections which are independent circular
// buffers over which tasks are distributed. For pops, the sections map to
// blocks in a round-robin pattern; block_id = section_id % _num_blocks. For
// pushes, we have a global _write_section variable which is atomically
// incremented so that the next writer writes the next section. push() and pop()
// are warp-local and must be called by the entire warp. Each operation is
// blocking, ie. a section cannot be read from and written to concurrently.
// push() supports multiple tasks per thread. This is to avoid choosing a
// framework just yet (eg. Gunrock's FILTER step could simplify this)
template <typename TaskType>
class InitialScheduler {
public:
  // Common methods

  // Initialize the scheduler on host.
  unsigned init(const unsigned byte_capacity, const unsigned num_blocks,
                const unsigned num_warps);

  // Allocate and setup the scheduler on device.
  InitialScheduler<TaskType>* device_setup(const cudaDeviceProp& prop);

  // Dealloc the scheduler on device.
  unsigned device_dealloc();

  // Push a warp's tasks to the queue. 
  // MUST be called by all threads in a warp (syncs across all threads in warp).
  // tasks is a thread-local buffer of num_tasks
  __device__ PushResult push(TaskType* tasks, unsigned num_tasks,
                             const unsigned lane_id);

  // Pop tasks from the queue for a warp's threads to work on. 
  // MUST be called by all threads in a warp. 
  // dst is a thread-local destination buffer. At most
  // one task per thread will be popped, up to warp_max_tasks across the warp.
  // tasks_popped is the number of tasks popped across the warp; when this is
  // <32, we pop into the lowest lanes.
  // TODO: make tasks_popped thread-local (boolean)
  __device__ PopResult pop(TaskType* dst, unsigned* tasks_popped,
                           const unsigned warp_max_tasks,
                           const unsigned block_id, const unsigned lane_id);

private:
  // Common attributes
  unsigned _on_device;
  unsigned _byte_capacity;
  TaskType* _queue;
  unsigned _queue_capacity_bytes;
  char* _d_alloc;

  // Implementation-specific attributes
  struct SectionMD {
    unsigned resv_start;
    unsigned resv_end;
    unsigned busy; // bool
    unsigned full; // bool
  };

  struct SectionMD* _section_metadata;
  unsigned _num_warps;
  unsigned _num_blocks;
  unsigned _num_sections;
  unsigned _section_capacity;
  unsigned _write_section;
  unsigned _num_full_sections;
  unsigned _num_warps_writing;
  unsigned _num_empty_sections;
  unsigned _num_warps_reading;
};

#include <scheduler/initial_scheduler.cut>
