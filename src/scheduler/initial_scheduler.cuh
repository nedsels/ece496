#pragma once

#include <graph/csr_graph.cuh>
#include <scheduler/enum.cuh>

// HYPERPARAMTERS
#define NUM_SECTIONS 1

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
  __device__ PushResult push(TaskType* tasks, unsigned num_tasks,
                             const unsigned thread_id);

  // Pop tasks from the queue for a warp's threads to work on.
  __device__ PopResult pop(TaskType* dst, unsigned* tasks_read,
                           const unsigned max_tasks, const unsigned block_id,
                           const unsigned thread_id);

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
  unsigned _num_non_empty_sections;
  unsigned _num_warps_reading;
};

#include <scheduler/initial_scheduler.cut>
