#pragma once

// Layout of a warp's shared memory
template<size_t CAP, typename Task, typename WorkItem>
struct alignas(16) WarpSMem {
    // Work pool: one slot per lane
    WorkItem work_pool[32];
    // Buffer to stage new tasks before commiting to scheduler
    Task stage[CAP];
};
