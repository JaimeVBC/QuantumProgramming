#include "header.h"
#include <string>
#include <fstream>
#include <vector>
#include <utility> // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
//#include "sm_60_atomic_functions.h"


#define MAX_THREADS 1024
#define MAX_BLOCKS 30
#define MAX_PERMS 5041

#define CUDA_RUN(x_) {cudaError_t cudaStatus = x_; if (cudaStatus != cudaSuccess) {fprintf(stderr, "Error  %d - %s\n", cudaStatus, cudaGetErrorString(cudaStatus)); goto Error;}}
#define SAFE(x_) {if((x_) == NULL) printf("NullPtr Error: possible out of memory.\n Error detected in line: %d\n", __LINE__);}


#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#define cuda_SYNCTHREADS() __syncthreads()
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#define cuda_SYNCTHREADS()
#endif

#ifdef __CUDACC__
#define cuda_SYNCTHREADS() __syncthreads()
#else
#define cuda_SYNCTHREADS()
#endif


__device__ __shared__ int32_t shared_cost;



// Quick implementation of factorial of a number
__host__ unsigned long long factorial(int32_t n)
{
	unsigned long long result = 1;
	for (int i = 1; i <= n; i++)
		result = result * i;

	return result;
}

int main(int argc, char *argv[])
{

	if (argc < 2)  // Two arguments need to be specified at least. Name of the programm itself and number of nodes to compute.
	{	printf("Number of nodes must be specified");
		return 0;
	}
	int size8 = sizeof(int8_t);
	int size32 = sizeof(int32_t);
	unsigned long long total_permutations, thread_perms, num_blocks = 1, num_threads, num_kernels = 1;
	float elapsed_time;
	cudaEvent_t startEvent, stopEvent;

	// Unified memory
	//float* shared_cost;

	// Host variables 
	int8_t * nodes, *shortestPath, *selected, selectedK = 0;
	int32_t numNodes = atoi(argv[1]), *cost; //Number of cities to be computed
	int32_t* edgesWeights;

	unsigned long long threads_per_kernel;

	// Device variables 
	int8_t * d_nodes, *d_shortestPath, *d_selected, * d_selectedK;
	int32_t *d_numNodes, *d_costs;
	int32_t* d_edgesWeights;
	unsigned long long* d_threads_per_kernel; 

	total_permutations = factorial(numNodes - 1); // Number of combinations to be computed is (N-1)! where is N is the number of nodes.
	printf("%d nodes results in %llu combinations\n", numNodes - 1, total_permutations);

	// Calculation of what is the max number of permutations per thread possible without exceeding MAX_PERMS
	for (selectedK = 1; selectedK < numNodes - 2; selectedK++)
	{
		thread_perms = factorial(numNodes - 1 - selectedK);
		if (thread_perms < MAX_PERMS) break;
	}
	// Calculation of how many threads do we need based on the permutations per thread and the total number of permutations to be processed.
	num_threads = total_permutations / thread_perms;
	
	// If threads exceed the maximum, they will be equally distributed in different blocks
	int k;
	while (num_threads > MAX_THREADS) {
		k = 2;
		while (num_threads % k != 0) k++;
		num_threads /= k;
		num_blocks *= k;
	}
	// If blocks exceed the maximum, they will be equally distributed in different kernels
	while (num_blocks > MAX_BLOCKS) {
		k = 2;
		while (num_blocks % k != 0) k++;
		num_blocks /= k;
		num_kernels *= k;
	}
	threads_per_kernel = num_blocks * num_threads;
	
	// Print problem configuration
	printf("K selected: %d\n", selectedK);
	printf("Num_threads: %llu\n Thread_perms: %llu\n Num_blocks %llu\n Num_kernels %llu\n Threads_per_kernel: %llu\n", num_threads, thread_perms, num_blocks, num_kernels, threads_per_kernel);


	dim3 block_dim(num_threads, 1, 1);
	dim3 grid_dim(num_blocks, 1, 1);

	// Memory allocations with SAFE macro in case one of them fails due to memory not being able.
	SAFE(nodes = (int8_t *)malloc(numNodes * size8));
	SAFE(shortestPath = (int8_t *)calloc(num_blocks * numNodes, size8));
	SAFE(edgesWeights = (int32_t*)malloc(numNodes * sizeof(int32_t) * numNodes));
	SAFE(cost = (int32_t*)calloc(num_blocks * numNodes, sizeof(int32_t)));
	SAFE(selected = (int8_t *)malloc(threads_per_kernel * numNodes * size8));

	// Device memory allocation for data in the device (GPU)
	CUDA_RUN(cudaMalloc((void **)&d_nodes, numNodes * size8));
	CUDA_RUN(cudaMalloc((void **)&d_shortestPath, numNodes * size8 * num_blocks));
	CUDA_RUN(cudaMalloc((void **)&d_edgesWeights, numNodes * sizeof(int32_t) * numNodes));
	CUDA_RUN(cudaMalloc((void **)&d_costs, num_blocks * sizeof(int32_t)));
	CUDA_RUN(cudaMalloc((void **)&d_numNodes, size32));
	CUDA_RUN(cudaMalloc((void **)&d_selectedK, size8));
	CUDA_RUN(cudaMalloc((void **)&d_selected, threads_per_kernel * numNodes * size8));
	CUDA_RUN(cudaMalloc((void **)&d_threads_per_kernel, sizeof(unsigned long long)));
	

	srand(time(NULL));
	initialize(nodes, edgesWeights, numNodes);

	// Translation of the data from host (CPU) to the device (GPU)
	CUDA_RUN(cudaMemcpy(d_nodes, nodes, numNodes * size8, cudaMemcpyHostToDevice));
	CUDA_RUN(cudaMemcpy(d_shortestPath, shortestPath, numNodes * size8 * num_blocks, cudaMemcpyHostToDevice));
	CUDA_RUN(cudaMemcpy(d_edgesWeights, edgesWeights, numNodes * sizeof(int32_t) * numNodes, cudaMemcpyHostToDevice));
	CUDA_RUN(cudaMemcpy(d_numNodes, &numNodes, size32, cudaMemcpyHostToDevice));
	CUDA_RUN(cudaMemcpy(d_selectedK, &selectedK, size8, cudaMemcpyHostToDevice));
	CUDA_RUN(cudaMemcpy(d_selected, selected, threads_per_kernel * numNodes * size8, cudaMemcpyHostToDevice));
	CUDA_RUN(cudaMemcpy(d_threads_per_kernel, &threads_per_kernel, sizeof(unsigned long long), cudaMemcpyHostToDevice));
	CUDA_RUN(cudaMemcpy(d_costs, cost, num_blocks * sizeof(int32_t), cudaMemcpyHostToDevice));

	// Creation of time events to measure times
	CUDA_RUN(cudaEventCreate(&startEvent));
	CUDA_RUN(cudaEventCreate(&stopEvent));
	CUDA_RUN(cudaEventRecord(startEvent, 0));
	
	// Kernels launching one by one
	float percentage;
	for (int i = 0; i < num_kernels; i++) {
		
		find_permutations_for_threads KERNEL_ARGS2( 1, 1 )(d_nodes, d_selectedK, d_selected, d_numNodes, d_threads_per_kernel);
		CUDA_RUN(cudaGetLastError());
		CUDA_RUN(cudaDeviceSynchronize());
		
		combinations_kernel KERNEL_ARGS2(grid_dim, block_dim) (d_selected, d_selectedK, d_shortestPath, d_edgesWeights, d_costs, d_numNodes);
		CUDA_RUN(cudaGetLastError());
		CUDA_RUN(cudaDeviceSynchronize());
		
		

		// Printing progress out in the console 
		percentage = (100. / (float) num_kernels * (float)(i + 1));
		printf("\rProgress : ");
		for (int j = 0; j < 10; j++) {
			if ((percentage / 10) / j > 1) printf("#");
			else printf(" ");
		}
		printf(" [%.2f%%]", percentage);
		fflush(stdout); // Refresh the output to update the "animation"
	}
	CUDA_RUN(cudaEventRecord(stopEvent, 0));
	CUDA_RUN(cudaEventSynchronize(stopEvent));
	CUDA_RUN(cudaEventElapsedTime(&elapsed_time, startEvent, stopEvent));
	CUDA_RUN(cudaMemcpy(shortestPath, d_shortestPath, num_blocks * numNodes * size8, cudaMemcpyDeviceToHost));
	CUDA_RUN(cudaMemcpy(cost, d_costs, num_blocks * sizeof(int32_t), cudaMemcpyDeviceToHost));
	
	CUDA_RUN(cudaDeviceSynchronize());

	printf("\nTime passed:  %3.1f ms \n", elapsed_time);
	print_Graph(edgesWeights, numNodes);

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
		print_ShortestPath(&shortestPath[index * numNodes], min, numNodes);
	}

	free(nodes);
	free(shortestPath);
	free(edgesWeights);
	free(cost);
	free(selected);

	cudaFree(d_nodes);
	cudaFree(d_shortestPath);
	cudaFree(d_edgesWeights);
	cudaFree(d_costs);
	cudaFree(d_numNodes);
	cudaFree(d_selectedK);
	cudaFree(d_selected);
	cudaFree(d_threads_per_kernel);

	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);

	getchar();

	return 0;

Error:
	free(nodes);
	free(shortestPath);
	free(edgesWeights);
	free(cost);
	free(selected);

	cudaFree(d_nodes);
	cudaFree(d_shortestPath);
	cudaFree(d_edgesWeights);
	cudaFree(d_costs);
	cudaFree(d_numNodes);
	cudaFree(d_selectedK);
	cudaFree(d_selected);
	cudaFree(d_threads_per_kernel);

	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);

	getchar();

	return 0;
}

__global__
void find_permutations_for_threads(int8_t * nodes_ids, int8_t * k, int8_t * choices, int32_t * size, unsigned long long * threads_per_kernel) {
	int32_t length = *size;
	int8_t index = 1;
	unsigned long long count = 0;
	for (count = 0; count < *threads_per_kernel; count++) {
		for (int i = 0; i < length; i++) {
			choices[i + count * length] = nodes_ids[i];
		}
		reverse(nodes_ids + *k + index, nodes_ids + length);
		next_permutation(nodes_ids + index, nodes_ids + length);
	}
}

__global__
void combinations_kernel(int8_t * choices, int8_t * k, int8_t * shortestPath, int32_t* graphWeights, int32_t* cost, int32_t * size) {
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t length = *size;
	int8_t index = 1;

	/* local variables */
	int8_t * _path, *_shortestPath;
	int32_t _tcost;

	SAFE(_path = (int8_t *)malloc(length * sizeof(int8_t)));
	SAFE(_shortestPath = (int8_t *)malloc(length * sizeof(int8_t)));
	
	_tcost = length * 100;


	memcpy(_path, choices + tid * length, length * sizeof(int8_t));
	memcpy(_shortestPath, shortestPath, length * sizeof(int8_t));
	
	if (threadIdx.x == 0) {
		if (cost[blockIdx.x] == 0) cost[blockIdx.x] = length * 100;
		shared_cost = length * 100;
	}

	cuda_SYNCTHREADS();
		
	
	do {
		copy_array2(_path, _shortestPath, &_tcost, graphWeights, length, tid);
	} while (next_permutation(_path + *k + index, _path + length));
	
	
	if (_tcost == shared_cost) {
		//printf("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
		atomicMin(&cost[blockIdx.x], _tcost);
		if (cost[blockIdx.x] == _tcost) {
			memcpy(shortestPath + blockIdx.x * length, _shortestPath, length * sizeof(int8_t));
		}
	}

	free(_path);
	free(_shortestPath);
}

__host__
void initialize(int8_t * nodes_ids, int32_t* nodesDistances, int32_t size)
{
	std::ifstream file("NodesDistances.csv");

	for (int row = 0; row < size; ++row)
	{
		std::string line;
		std::getline(file, line);
		if (!file.good())
			break;

		std::stringstream iss(line);

		for (int col = 0; col < size; ++col)
		{
			std::string val;
			std::getline(iss, val, ';');
			if (!iss.good())
				break;

			std::stringstream convertor(val);
			float temp;
			convertor >> temp;
			//printf("%f\t",temp);
			nodesDistances[row*size+col] = (int32_t)(temp*10000);
			//printf("%d\t", nodesDistances[row * size + col]);
		}
		printf("\n");
	}
}

__host__
void print_Graph(int32_t* nodesDistances, int32_t size) {
	int i, j;
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			printf("%3.8f\t", (float)(nodesDistances[i*size+j]*0.0001));
		}
		printf("\n");
	}
}

__host__
void print_ShortestPath(int8_t * shortestPath, int32_t cost, int32_t size) {
	int i;
	if (cost == (size * 100)) printf("no possible path found.Cost is %d and size is %d\n",cost,size);
	else {
		for (i = 0; i < size; i++) {
			printf("%d\t", shortestPath[i]);
		}
		printf("\nCost: %f\n", cost*0.0001);
	}
}

__device__
void swap(int8_t *x, int8_t *y) { int8_t tmp = *x; *x = *y;	*y = tmp; }

__device__
void reverse(int8_t *first, int8_t *last) { while ((first != last) && (first != --last)) swap(first++, last); }

__device__
void copy_array2(int8_t * path, int8_t * shortestPath, int32_t* tcost, int32_t* weights, int8_t length, int tid) {
	int32_t sum = 0;
	for (int32_t i = 0; i < length; i++) {
		int32_t val = weights[path[i] * length + path[(i + 1) % length]];
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

__device__
bool next_permutation(int8_t * first, int8_t * last) {
	if (first == last) return false;
	int8_t * i = first;
	++i;
	if (i == last) return false;
	i = last;
	--i;

	for (;;) {
		int8_t * ii = i--;
		if (*i < *ii) {
			int8_t * j = last;
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
/*__device__ static float atomicMinOwn(float* address, float val)
{
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}*/