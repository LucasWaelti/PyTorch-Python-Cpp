# PyTorch-Python-Cpp
Experimental repository compiling comparison implementation of code written in Python and C++.
While the Python implementations should run on any OS, the C++ implementations were only tested on Linux. 

## Get PyTorch/libtorch
Go to [PyTorch](https://pytorch.org/) and download the corresponding package. 

![Screenshot from 2019-09-21 15-54-55](/home/lucaswaelti/Pictures/Screenshot from 2019-09-21 15-54-55.png)

Then simply extract the zip file where you want to install the library or follow the instruction depending on the language selected. 

## ./example-app
This folder contains the implemented *minimal example* provided by PyTorch. It can be found [here](https://pytorch.org/cppdocs/installing.html). 

The project can be compiled and run like so: 
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
make
```

## ./pycpp

This folder aims at implementing the same program both in **C++** and **Python** to illustrate the similarities of the **PyTorch** API between both languages. 

### Python 

To run the python script: 

```bash
python3 -m demo
```

### C++ 

To build the C++ code: 

```bash
./build.sh
```

> :warning: you need to change the **libtorch** installation path in the `build.sh` script. 

To run the C++ code: 

```bash
./run.sh
```

