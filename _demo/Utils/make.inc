CXX = g++ -fPIC -fopenmp
MPXX = mpicxx -fPIC -fopenmp
CUDAXX = nvcc -shared -Xcompiler -fPIC -c
CXXFLAGS = -O3 -Wno-div-by-zero -shared -std=c++17 `python3 -m pybind11 --includes`
EIGEN = -I /HOME/scw6do2/run/soft/eigen-3.4.0/install/include/eigen3 
ASTRA = -DASTRA_CUDA -I /HOME/scw6do2/run/soft/tomo_TV/thirdparty/astra-toolbox -I /HOME/scw6do2/run/soft/tomo_TV/thirdparty/astra-toolbox/install/include
ASTRA_LIB = -L /HOME/scw6do2/run/soft/tomo_TV/thirdparty/astra-toolbox/install/lib -lastra 
CUDA = -I /data/apps/cuda/11.8/include -L /data/apps/cuda/11.8/lib64 -lcudart -lz
