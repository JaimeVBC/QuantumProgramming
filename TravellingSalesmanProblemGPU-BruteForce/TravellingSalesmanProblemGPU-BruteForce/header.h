#ifndef __COMBINATIONS_H__
#define __COMBINATIONS_H__

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <string>
#include <stdint.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// __device__ void * cudaMemmove(void * dst0, const void * src0, register size_t length);
__device__ void swap(int8_t *x, int8_t *y);
__device__ void reverse(int8_t *first, int8_t *last);
__device__ void copy_array(int8_t * _path, int8_t *_shortestPath, float * _tcost, float ** weights, int8_t length, int tid, float * shared_cost);
__device__ bool next_permutation(int8_t * first, int8_t * last);
__device__ float atomicMinOwn(float* address, float val);
__global__ void find_permutations_for_threads(int8_t * city_ids, int8_t * k, int8_t * choices, int32_t * size, unsigned long long * perm_counter);
__global__ void combinations_kernel(int8_t * choices, int8_t * k, int8_t * shortestPath, float ** graphWeights, float * cost, int32_t * size, float * shared_cost);
__host__ void initialize(int8_t * city_ids, float ** graphWeights, int32_t size);
__host__ void print_Graph(float ** graphWeights, int32_t size);
__host__ void print_ShortestPath(int8_t * shortestPath, float cost, int32_t size);
__host__ unsigned long long factorial(int32_t n);

#endif
