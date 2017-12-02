#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/
#include <stdio.h> // stdio functions are used since C++ streams aren't necessarily thread safe
#include <stdlib.h>
#include <string.h>
#include "graphio.h"
#include "graph.h"
#ifdef __cplusplus
}
#endif /*__cplusplus*/

#include <string>

//#include <omp.h>

//#define DEBUG
#define NREPS 10 // number of repetations for time calculations

#define THREADS_PER_BLOCK 1024

__global__ void ClosenessCentKernel(int *result, const etype *rowPtr, const vtype *colInd, vtype nov)
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if (index < nov)
		result[index] = -1;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t CudaClosenessCent(int *result, const etype *rowPtr, const vtype *colInd, vtype nov)
{
	etype *dev_rowPtr = 0;
	vtype *dev_colInd = 0;
	int *dev_result = 0;
	cudaError_t cudaStatus;

	//===========================================================================================================================
	// Allocate GPU buffers for three vectors (two input, one output)
	cudaStatus = cudaMalloc((void**)&dev_result, nov * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_rowPtr, nov * sizeof(etype));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_colInd, rowPtr[nov] * sizeof(vtype));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//===========================================================================================================================
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_rowPtr, rowPtr, nov * sizeof(etype), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_colInd, colInd, rowPtr[nov] * sizeof(vtype), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//===========================================================================================================================
	// Launch a kernel on the GPU with one thread for each element, and check for errors.
	int numThreads = (int)sqrt(THREADS_PER_BLOCK);
	dim3 dimBlock(numThreads, numThreads, 1);
	//dim3 dimGrid(nov / numThreads, nov / numThreads);
	printf("%d, %d\n", nov, nov / numThreads);
	ClosenessCentKernel<<<(nov+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, dimBlock>>>(dev_result, dev_rowPtr, dev_colInd, nov);

	//===========================================================================================================================
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(result, dev_result, nov * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_result);
	cudaFree(dev_rowPtr);
	cudaFree(dev_colInd);

	return cudaStatus;
}
