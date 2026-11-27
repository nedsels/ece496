#pragma once

#include <scheduler/enum.cuh>
#include <scheduler/initial_scheduler.cuh>

#include <variant>

template <typename TaskType>
using Scheduler = std::variant<InitialScheduler<TaskType>>;

template <typename TaskType>
inline Scheduler<TaskType> get_scheduler(SchedulerType sched_type) {
  if (sched_type == INITIAL_SCHEDULER) {
    return InitialScheduler<TaskType>();
  }
  throw std::invalid_argument("Unknown scheduler type");
}

// Takes host scheduler and returns setup device scheduler
// NOTE: This is TOTAL number of warps
template<typename Scheduler>
Scheduler* setup_scheduler(
  Scheduler& h_sched, const unsigned num_blocks, 
  const unsigned num_warps, const cudaDeviceProp& prop
) {
  // HYPERPARAMETER
  // Reserve 1/8 of free VRAM (completely arbitrary)
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  const unsigned scheduler_byte_capacity =
      static_cast<unsigned>(free_bytes >> 3);

  h_sched.init(scheduler_byte_capacity, num_blocks, num_warps);
  return h_sched.device_setup(prop);
}
