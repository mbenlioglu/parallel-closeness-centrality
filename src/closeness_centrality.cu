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

void PrintUsage(const char *appName)
{
	printf("./%s <graph_location> <GPU deviceID0\n>", appName);
}

int main(int argc, char *argv[])
{
	std::string baseName = std::string(argv[0]);
	std::string fillerAsterisk(100, '*');
	std::string fillerDashes(100, '-');

	// Get executable name from path
#ifdef WIN32
	baseName = baseName.substr(baseName.rfind('\\') + 1);
#else
	baseName = baseName.substr(baseName.rfind('/') + 1);
#endif // WIN32

	printf((fillerAsterisk + "\n").c_str());
	printf("Starting %s ...\n", baseName.c_str());
	printf((fillerAsterisk + "\n").c_str());

	printf("\nInitializing Device...\n");
	
	//===========================================================================================================================
	// set the CUDA capable GPU to be used
	//
	int num_gpus = 0;   // number of CUDA GPUs
	int devID = atoi(argv[2]); // selected device id
	
	cudaGetDeviceCount(&num_gpus);

	if (num_gpus < 1)
	{
		printf("no CUDA capable devices were detected\n");
		return EXIT_FAILURE;
	}
	else if (devID > num_gpus || devID < 0)
	{
		printf("Invalid device ID\n");
		return EXIT_FAILURE;
	}

	cudaDeviceProp dprop;
	cudaError_t cudaStatus = cudaGetDeviceProperties(&dprop, devID);
	printf("   %s #%d: %s\n\n","Selected Device", devID, dprop.name);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	cudaSetDevice(devID);

	//===========================================================================================================================
	// read Graph
	//
	etype *rowPtr;
	vtype *colInd;
	ewtype *ewghts;
	vwtype *vwghts;
	vtype nov;

	printf("Reading graph... ");
	// Graph reading
	if (argc < 3)
	{
		PrintUsage(baseName.c_str());
		return EXIT_FAILURE;
	}

	if (read_graph(argv[1], &rowPtr, &colInd, &ewghts, &vwghts, &nov, 0) == -1)
	{
		printf("error in graph read\n");
		return EXIT_FAILURE;
	}
	printf("done.\n");
	printf((fillerDashes + "\n").c_str());
	//===========================================================================================================================

	int *result = new int[nov]();

    // Calculate closeness-centrality in parallel.
    cudaStatus = CudaClosenessCent(result, rowPtr, colInd, nov);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaClosenessCent failed!");
        return EXIT_FAILURE;
    }

	for (size_t i = 0; i < nov; i++)
	{
		if (result[i] != -1)
		{
			printf("WRONG! %d=%d\n", i, result[i]);
			break;
		}
	}

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
