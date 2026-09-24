#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../src/iter.cuh"
#include <doctest/doctest.h>
#include <vector>

using namespace iter;

// Test basic construction and access
TEST_CASE("SliceIterator: Basic Construction") {
    std::vector<u32> data = {1, 2, 3, 4, 5};
    auto iter = SliceIterator<u32>(data.data(), 3, 0, 5);
    
    CHECK_EQ(*iter, 1);
    CHECK_EQ(iter[0], 1);
    CHECK_EQ(iter[1], 2);
    CHECK_EQ(iter[2], 3);
}

// Test offsets and wraparound
TEST_CASE("SliceIterator: Offset and Wrap") {
    std::vector<u32> data = {1, 2, 3, 4, 5};
    auto iter = SliceIterator<u32>(data.data(), 3, 3, 5);
    
    CHECK_EQ(*iter, 4);
    CHECK_EQ(iter[0], 4);
    CHECK_EQ(iter[1], 5);
    CHECK_EQ(iter[2], 1);
}

// Test iterator operations
TEST_CASE("SliceIterator: Iterator Operations") {
    std::vector<u32> data = {1, 2, 3, 4, 5};
    auto iter = SliceIterator<u32>(data.data(), 5, 0, 5);
    
    // Test the ++ operator
    CHECK_EQ(*iter, 1);
    ++iter;
    CHECK_EQ(*iter, 2);
    iter++;
    CHECK_EQ(*iter, 3);
    
    // Test the -- operator
    --iter;
    CHECK_EQ(*iter, 2);
    iter--;
    CHECK_EQ(*iter, 1);
    
    // Test the += and -= operators
    iter += 2;
    CHECK_EQ(*iter, 3);
    iter -= 1;
    CHECK_EQ(*iter, 2);
}

// Test comparison operators
TEST_CASE("SliceIterator: Comparison Operators") {
    std::vector<u32> data = {1, 2, 3, 4, 5};
    auto iter1 = SliceIterator<u32>(data.data(), 5, 0, 5);
    auto iter2 = SliceIterator<u32>(data.data(), 5, 0, 5);
    auto iter3 = SliceIterator<u32>(data.data(), 5, 1, 5);
    
    CHECK(iter1 == iter2);
    CHECK_FALSE(iter1 != iter2);
    CHECK(iter1 < iter3);
    CHECK(iter3 > iter1);
    CHECK(iter1 <= iter2);
    CHECK(iter1 >= iter2);
}

// Test conversion to const
TEST_CASE("SliceIterator: Const Conversion") {
    std::vector<u32> data = {1, 2, 3, 4, 5};
    SliceIterator<u32> non_const_iter(data.data(), 5, 0, 5);
    SliceIterator<const u32> const_iter = non_const_iter; // Implicit conversion
    
    CHECK_EQ(*const_iter, 1);
    CHECK_EQ(const_iter[2], 3);
}

// Test the `make_slice_iter` function
TEST_CASE("SliceIterator: Make Slice Iter") {
    std::vector<u32> data = {1, 2, 3, 4, 5};
    
    // Test the non-const version
    PolyPtr ptr {
        .ptr = data.data(),
        .len = 5,
        .rotate = 2,
        .offset = 0,
        .whole_len = 5
    };
    
    auto iter = make_slice_iter<u32>(ptr);
    CHECK_EQ(*iter, 4);
    CHECK_EQ(iter[0], 4);
    CHECK_EQ(iter[1], 5);
    CHECK_EQ(iter[2], 1);
    
    // Test the const version
    ConstPolyPtr const_ptr {
        .ptr = data.data(),
        .len = 5,
        .rotate = 2,
        .offset = 0,
        .whole_len = 5
    };
    
    auto const_iter = make_slice_iter<u32>(const_ptr);
    CHECK_EQ(*const_iter, 4);
    CHECK_EQ(const_iter[0], 4);
    CHECK_EQ(const_iter[1], 5);
    CHECK_EQ(const_iter[2], 1);
}

// Test iterator difference calculation
TEST_CASE("SliceIterator: Iterator Difference") {
    std::vector<u32> data = {1, 2, 3, 4, 5};
    auto iter1 = SliceIterator<u32>(data.data(), 5, 1, 5);
    auto iter2 = SliceIterator<u32>(data.data(), 5, 3, 5);
    
    CHECK_EQ(iter2 - iter1, 2);
    CHECK_EQ(iter1 - iter2, 3); // Difference across wraparound
}
