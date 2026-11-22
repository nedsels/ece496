#pragma once

template<typename Item>
struct RingBuffer {
    // This is expected to be dynamically allocated but can change this to be static
    volatile Item *buffer;
    unsigned head;
    unsigned tail;
    unsigned size; /* Must be power of two */

    __device__ void push(Item item) {
        // Wait while full
        while (head - tail == size) {}
        buffer[head & (size - 1)] = item;
        // Do memory fence to ensure write completes before publishing
        __threadfence_block();
        head += 1;
    }

    __device__ Item pop() {
        // Wait while empty
        while (head == tail) {}
        Item item = buffer[tail & (size - 1)];
        // Ensure read completes before freeing tail slot
        __threadfence_block();
        tail += 1;
        return item;
    }

    __host__ __device__ void init(volatile Item *data, const unsigned s) {
        buffer = data;
        head = 0;
        tail = 0;
        size = s;
    }
}