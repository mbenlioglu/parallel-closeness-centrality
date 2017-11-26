# GPU Asisted Parallelization of Closeness-Centrality Calculation
This program executes sequential and parallel versions of closeness-centrality calculation, outputs the execution
times of sequential and parallel versions(with 1, 2, 4, 8, 16 threads) 

**Implemented by:**

 * [M.Mucahid Benlioglu](https://github.com/mbenlioglu)


## Getting Started
There are two Makefiles and a Python script included this project in order to crawl the data used for developement and
testing (Linux/Windows), and to compile sources into executable (Linux).

### Data Crawl

**For Linux:**

One of the Makefiles, `./Makefile-data`, is to crawl the data that is used for testing and developement. Running following
command on terminal will download and extract the data into `./data/`. Note that, this folder will be created if it does not
already exists.

    $ make -f Makefile-data

Also note that, after crawling and extracting data, following command can be run to get rid of downloaded archive files (Not
the extracted data folders/files just archives).

    $ make -f Makefile-data clean

**For Windows:**

Since Makefiles and command-line tools used in the Makeflie (e.g. tar) is not officially supported in Windows, we have
provided a script written in [Python 2.7.x](https://docs.python.org/2/), to download and extract the data. Note that,
the data is archived with TAR & GZ compression, where there is also no official tools from Windows to decompress.
Therefore, we have used [WinRAR](https://www.win-rar.com/) to decompress these archieves in the script, which you may
need to download if you do not have. Running the following script will download and extract the data into `./data/` folder,
and will create this folder if it doesn't already exists.
    
    $ python ./DataCrawl-windows.py

Note that, running this script with no parameters will assume that, `"C:/Program Files/WinRAR/WinRAR.exe"` exists. If
WinRAR is installed in some other location, it can be specified with `--winrar_path` flag. Also note that giving `-r` or
`--remove-archieves` flag will remove original archives after donwloading and extraction finishes (similar to `make clean`)_.
For more information about this script, execute following command in project root.

    $ python ./DataCrawl-windows.py -h
------
Input data is in [MTX format](http://math.nist.gov/MatrixMarket/formats.html) and converted into [CRS format](http://netlib.org/linalg/html_templates/node91.html)
for processing. For more information about the data used in this project visit [SuiteSparse Matrix Collection](https://sparse.tamu.edu/).

### Compiling

The other Makefile, `./Makefile` is to compile codes into executable in Linux environment. Executing following command
in the project root will create an executable named "coloring" which takes single input argument, which is the path to
the dataset.

    $ make

For Windows use Visual Studio with provided *.sln & *.vxproj files and use /openmp flag to compile the codes.

The program will run sequential and parallel versions of the algorithm and output the execution time and number of colors used.


## Algorithm & Implementation
In this section, sequential and parallel version of the algorithm will be explained. Performance results of each algorithm can
be found in results section.



## Results
In this section, result tables containing various information about execution results is given.



