#include <scheduler/enum.cuh>

SchedulerType from_string(std::string sched_str) {
  if (sched_str == "initial_scheduler") {
    return INITIAL_SCHEDULER;
  }
  throw std::invalid_argument("Unknown scheduler type");
}