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
