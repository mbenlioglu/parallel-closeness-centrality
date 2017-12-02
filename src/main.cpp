#include <cuda_runtime.h>

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

cudaError_t CudaClosenessCent(int *result, const etype *rowPtr, const vtype *colInd, vtype nov);

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
	printf("   %s #%d: %s\n\n", "Selected Device", devID, dprop.name);
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