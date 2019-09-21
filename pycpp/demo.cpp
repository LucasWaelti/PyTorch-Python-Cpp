#include <iostream>
#include <torch/torch.h>

#define BASICS 
#define LEARN
#define DIM 100 // Dataset size

torch::Tensor create2DDataSet(unsigned int dim=DIM){
  return torch::rand({dim,2});
}

torch::Tensor createGroundTruth(const torch::Tensor& input){
  torch::Tensor out = torch::empty(input.size(0));
  for(int i=0; i<input.size(0); i++){
    return torch::rand(input.size(0));
  }
}

int main() {
  std::cout << "Running C++ implementation\n" << std::endl;

#ifdef BASICS
  std::cout << "Basic Tensor manipulations:\n" << std::endl;

  std::cout << "Manually initialized tensor:" << std::endl;
  torch::Tensor t = torch::tensor({1,2,3,4});
  t = torch::_unsafe_view(t,{2,-1});
  std::cout << t << std::endl << std::endl;

  std::cout << "Transpose a tensor:" << std::endl;
  t = torch::transpose(t,0,1);
  std::cout << t << std::endl << std::endl;

  std::cout << "Concatenate tensors:" << std::endl;
  torch::Tensor t1 = torch::_unsafe_view(
    torch::tensor({5,6}),{-1,2});
  t = torch::cat({t,t1},0);
  std::cout << t << std::endl << std::endl; 

#elif LEARN
  // TODO 
#endif
}
