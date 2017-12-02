coloring: ./src/graphio.c ./src/mmio.c ./src/closeness_centrality.cpp ./src/closeness_centrality.cpp
	gcc ./src/graphio.c -c -O3
	gcc ./src/mmio.c -c -O3
	nvcc -O3 -c ./src/closeness_centrality.cu -Xcompiler -O3
	g++ -o closenessCentrality ./src/closeness_centrality.cpp closeness_centrality.o mmio.o graphio.o -lcuda -L/usr/local/cuda/lib64/ -fpermissive -fopenmp -O3 -std=c++14
clean:
	rm closenessCentrality *.o
