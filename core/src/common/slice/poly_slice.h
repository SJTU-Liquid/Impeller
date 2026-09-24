#pragma once
#include <cuda_runtime.h>

extern "C" struct PolySlice {
    unsigned long long begin, slice_len, len;

    __host__ __device__ unsigned long long to_origin(unsigned long long idx) const {
        assert(idx < slice_len);
        return (begin + idx) % len;
    }

    PolySlice(unsigned long long begin, unsigned long long slice_len, unsigned long long len)
        : begin(begin), slice_len(slice_len), len(len) {}
};