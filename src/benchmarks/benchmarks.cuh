
#pragma once
#include <optional>
/* 
 * Common benchmark code
 * TODO: Should put top level template that each problem inherits/uses  
*/

namespace Benchmarks{
    
enum class Status : int {
    SUCCESS = 0,
    ERROR = 1
};

// Different parameters set for the chosen benchmark
// Allows for common interface for all benchmarks
// Add more configurations as we support more kernels
struct BenchmarkConfig {
    std::optional<long> source;
    std::optional<long> destination;
};

} // end Benchmarks namespace

