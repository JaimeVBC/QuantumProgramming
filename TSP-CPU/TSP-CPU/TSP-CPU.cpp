#include "TSP-CPU.h"

#define SAFE(x_) {if((x_) == NULL) printf("out of memory. %d\n", __LINE__);}

int32_t shared_cost;

unsigned long long factorial(int32_t n) {
	int c;
	unsigned long long result = 1;

	for (c = 1; c <= n; c++)
		result = result * c;

	return result;
}

int gettimeofday(struct timeval* tv, struct timezone* tz)
{
	static LONGLONG birthunixhnsec = 116444736000000000;  /*in units of 100 ns */

	FILETIME systemtime;
	GetSystemTimeAsFileTime(&systemtime);

	ULARGE_INTEGER utime;
	utime.LowPart = systemtime.dwLowDateTime;
	utime.HighPart = systemtime.dwHighDateTime;

	ULARGE_INTEGER birthunix;
	birthunix.LowPart = (DWORD)birthunixhnsec;
	birthunix.HighPart = birthunixhnsec >> 32;

	LONGLONG usecs;
	usecs = (LONGLONG)((utime.QuadPart - birthunix.QuadPart) / 10);

	tv->tv_sec = (long long)(usecs / 1000000);
	tv->tv_usec = (long long)(usecs % 1000000);

	return 0;
}

int main(int argc, char* argv[]) {
	if (argc < 2) return 0;
	int size8 = sizeof(int8_t);
	int size32 = sizeof(int32_t);
	unsigned long long total_permutations;
	/* host variables */
	int8_t* city_ids, * shortestPath, * graphWeights, * choices;
	int32_t size = atoi(argv[1]), * cost;
	int8_t selected_K = 0;
	struct timeval startEvent;
	struct timeval stopEvent;

	total_permutations = factorial(size - 1);
	printf("factorial(%d): %llu\n", size - 1, total_permutations);

	SAFE(city_ids = (int8_t*)malloc(size * size8));
	SAFE(shortestPath = (int8_t*)calloc(num_blocks * size, size8));
	SAFE(graphWeights = (int8_t*)malloc(size * size8 * size));
	SAFE(cost = (int32_t*)calloc(num_blocks * size, size32));
	//SAFE(choices = (int8_t*)malloc(threads_per_kernel * size * size8));

	
	srand(time(NULL));
	initialize(city_ids, graphWeights, size);

	gettimeofday(&startEvent,NULL);

	float percentage;
	for (int i = 0; i < total_permutations; i++) {
		find_permutations_for_threads << < 1, 1 >> > (dev_city_ids, dev_selected_K, dev_choices, dev_size, dev_threads_per_kernel);
		CUDA_RUN(cudaGetLastError());
		CUDA_RUN(cudaDeviceSynchronize());
		combinations_kernel << < grid_dim, block_dim >> > (dev_choices, dev_selected_K, dev_shortestPath, dev_graphWeights, dev_cost, dev_size);
		CUDA_RUN(cudaGetLastError());
		CUDA_RUN(cudaDeviceSynchronize());
		percentage = (100. / (float)num_kernels * (float)(i + 1));
		printf("\rProgress : ");
		for (int j = 0; j < 10; j++) {
			if ((percentage / 10) / j > 1) printf("#");
			else printf(" ");
		}
		printf(" [%.2f%%]", percentage);
		fflush(stdout);
	}
	gettimeofday(&stopEvent,NULL);
	float time_passed = ((stopEvent.tv_sec - startEvent.tv_sec) * 1000000.0 + (stopEvent.tv_usec - startEvent.tv_usec)) / 1000.0));
	
	printf("\nTime passed:  %3.1f ms \n", time_passed);
	print_Graph(graphWeights, size);

	{
		int32_t min = cost[0];
		int8_t index = 0;
		for (int i = 1; i < num_blocks; i++) {
			if (cost[i] < min) {
				min = cost[i];
				index = i;
			}
		}
		printf("Shortest path found on block #%d:\n", index + 1);
		print_ShortestPath(&shortestPath[index * size], min, size);
	}

Error:
	free(city_ids);
	free(shortestPath);
	free(graphWeights);
	free(cost);
	free(choices);

	getchar();

	return 0;
}


void find_permutations_for_threads(int8_t* city_ids, int8_t* k, int8_t* choices, int32_t* size, unsigned long long* threads_per_kernel) {
	int32_t length = *size;
	int8_t index = 1;
	unsigned long long count = 0;
	for (count = 0; count < *threads_per_kernel; count++) {
		for (int i = 0; i < length; i++) {
			choices[i + count * length] = city_ids[i];
		}
		reverse(city_ids + *k + index, city_ids + length);
		next_permutation(city_ids + index, city_ids + length);
	}
}


void combinations_kernel(int8_t* choices, int8_t* k, int8_t* shortestPath, int8_t* graphWeights, int32_t* cost, int32_t* size) {
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t length = *size;
	int8_t index = 1;

	/* local variables */
	int8_t* _path, * _shortestPath;
	int32_t _tcost;

	SAFE(_path = (int8_t*)malloc(length * sizeof(int8_t)));
	SAFE(_shortestPath = (int8_t*)malloc(length * sizeof(int8_t)));
	_tcost = length * 100;

	memcpy(_path, choices + tid * length, length * sizeof(int8_t));
	memcpy(_shortestPath, shortestPath, length * sizeof(int8_t));

	if (threadIdx.x == 0) {
		if (cost[blockIdx.x] == 0) cost[blockIdx.x] = length * 100;
		shared_cost = length * 100;
	}

	__syncthreads();

	do {
		coppy_array(_path, _shortestPath, &_tcost, graphWeights, length, tid);
	} while (next_permutation(_path + *k + index, _path + length));

	if (_tcost == shared_cost) {
		atomicMin(&cost[blockIdx.x], _tcost);
		if (cost[blockIdx.x] == _tcost) {
			memcpy(shortestPath + blockIdx.x * length, _shortestPath, length * sizeof(int8_t));
		}
	}

	free(_path);
	free(_shortestPath);
}


void initialize(int8_t* city_ids, int8_t* graphWeights, int32_t size) {
	for (int i = 0; i < size; i++) {
		city_ids[i] = i;
		for (int j = 0; j < size; j++) {
			if (i == j)
				graphWeights[i * size + j] = 0;
			else
				graphWeights[i * size + j] = 99;
		}
	}

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size;) {
			int next = 1; // (rand() % 2) + 1;
			int road = rand() % 100 + 1;
			if (i == j) {
				j += next;
				continue;
			}
			graphWeights[i * size + j] = road;
			printf("%d\t", graphWeights[i * size + j]);
			j += next;
		}
	}

	for (int i = size - 1; i >= 0; i--) {
		graphWeights[((i + 1) % size) * size + i] = 1;
	}
}


void print_Graph(int8_t* graphWeights, int32_t size) {
	int i, j;
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			printf("%d\t", graphWeights[i * size + j]);
		}
		printf("\n");
	}
}


void print_ShortestPath(int8_t* shortestPath, int32_t cost, int32_t size) {
	int i;
	if (cost == (size * 100)) printf("no possible path found.\n");
	else {
		for (i = 0; i < size; i++) {
			printf("%d\t", shortestPath[i]);
		}
		printf("\nCost: %d\n", cost);
	}
}

void swap(int8_t* x, int8_t* y) { int8_t tmp = *x; *x = *y;	*y = tmp; }

void reverse(int8_t* first, int8_t* last) { while ((first != last) && (first != --last)) swap(first++, last); }

void coppy_array(int8_t* path, int8_t* shortestPath, int32_t* tcost, int8_t* weights, int8_t length, int tid) {
	int32_t sum = 0;
	for (int32_t i = 0; i < length; i++) {
		int8_t val = weights[path[i] * length + path[(i + 1) % length]];
		if (val == -1) return;
		sum += val;
	}
	if (sum == 0) return;
	atomicMin(&shared_cost, sum);
	if (shared_cost == sum) {
		*tcost = sum;
		memcpy(shortestPath, path, length * sizeof(int32_t));
	}
}

bool next_permutation(int8_t* first, int8_t* last) {
	if (first == last) return false;
	int8_t* i = first;
	++i;
	if (i == last) return false;
	i = last;
	--i;

	for (;;) {
		int8_t* ii = i--;
		if (*i < *ii) {
			int8_t* j = last;
			while (!(*i < *--j));
			swap(i, j);
			reverse(ii, last);
			return true;
		}
		if (i == first) {
			reverse(first, last);
			return false;
		}
	}
}