# PyTorch-Python-Cpp
Experimental repository compiling comparison implementation of code written in Python and C++.
While the Python implementations should run on any OS, the C++ implementations were only tested on Linux. 

## Get libtorch
Go to [PyTorch](https://pytorch.org/) and download the corresponding package. 

## ./example-app
This folder contains the implemented *minimal example* provided by PyTorch. It can be found [here](https://pytorch.org/cppdocs/installing.html). 

The project can be compiled and run like so: 
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
make
```

