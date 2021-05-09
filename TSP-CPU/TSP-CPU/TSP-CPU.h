#pragma once
#pragma once
#ifndef __COMBINATIONS_H__
#define __COMBINATIONS_H__

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <string>
#include <stdint.h>
#include <Windows.h>
#include <time.h>


// __device__ void * cudaMemmove(void * dst0, const void * src0, register size_t length);
void swap(int8_t* x, int8_t* y);
void reverse(int8_t* first, int8_t* last);
void coppy_array(int8_t* _path, int8_t* _shortestPath, int32_t* _tcost, int8_t* weights, int8_t length, int tid);
bool next_permutation(int8_t* first, int8_t* last);
void find_permutations_for_threads(int8_t* city_ids, int8_t* k, int8_t* choices, int32_t* size, unsigned long long* perm_counter);
void combinations_kernel(int8_t* choices, int8_t* k, int8_t* shortestPath, int8_t* graphWeights, int32_t* cost, int32_t* size);
void initialize(int8_t* city_ids, int8_t* graphWeights, int32_t size);
void print_Graph(int8_t* graphWeights, int32_t size);
void print_ShortestPath(int8_t* shortestPath, int32_t cost, int32_t size);
unsigned long long factorial(int32_t n);

#endif