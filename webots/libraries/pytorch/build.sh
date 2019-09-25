# Build the target 
mkdir build 
cd build
cmake -DCMAKE_PREFIX_PATH=/home/lucaswaelti/Documents/PyTorch/libtorch ..
make

# Move the generated shared object to the libraries folder of the Webots project
cp /home/lucaswaelti/Documents/PyTorch-Python-Cpp/webots/libraries/pytorch/build/libdemo.so /home/lucaswaelti/Documents/PyTorch-Python-Cpp/webots/libraries
