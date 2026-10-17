#pragma once

#include <stdexcept>
#include <string>

enum SchedulerType {
  INITIAL_SCHEDULER,
};

SchedulerType from_string(std::string sched_str);

enum PushResult {
  PUSH_SUCCESS,
  PUSH_FAILURE,
  PUSH_DEADLOCK,
};

enum PopResult {
  POP_SUCCESS,
  POP_FAILURE,
  POP_DONE,
};
