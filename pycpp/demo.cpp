#include <iostream>
#include <math.h>
#include <torch/torch.h>

//#define BASICS 
#define LEARN
#define DIM 10 // Dataset size


class Net: torch::nn::Module{
  public: 
    Net(){
      // Constructor - build the network's layers
      _in = register_module("in",torch::nn::Linear(2,10));
      _h = register_module("h",torch::nn::Linear(10,5));
      _out = register_module("out",torch::nn::Linear(5,1));
    }
    
    torch::Tensor forward(torch::Tensor x){
      // apply ReLU activations and sigmoid for the output
      x = torch::relu(_in->forward(x));
      x = torch::relu(_h->forward(x));
      x = torch::sigmoid(_out->forward(x));
        
      // return the output
      return x;
    }

  private:
    torch::nn::Linear _in{nullptr},_h{nullptr},_out{nullptr};
};

torch::Tensor create2DDataSet(unsigned int dim=DIM){
  return torch::rand({dim,2});
}

double rule(double x, double y, double r=0.5){
  // 0 if contained in circle with radius r, 
  // 1 if outside
  if(sqrt(pow(x,2) + pow(y,2)) < r){
    return 0.;
  }else{
    return 1.;
  }
}

torch::Tensor createGroundTruth(const torch::Tensor& input){
  torch::Tensor out = torch::empty(input.size(0));
  for(int i=0; i<input.size(0); i++){
    if(rule(input[i][0].item().toDouble(),
            input[i][1].item().toDouble()) == 0){
      out[i] = 0.;
    }else{
      out[i] = 1.;
    }
  }
  return out;
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

#elif defined(LEARN) 
  Net model = Net(); 

  torch::Tensor data  = create2DDataSet(DIM);
  torch::Tensor truth = createGroundTruth(data);
  std::cout << data  << std::endl;
  std::cout << truth << std::endl; 
  
#endif
  return EXIT_SUCCESS;
}
