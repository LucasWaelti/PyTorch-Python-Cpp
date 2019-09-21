#include <iostream>
#include <math.h>
#include <torch/torch.h>

//#define BASICS 
#define LEARN
#define DIM 100 // Dataset size


struct Net: torch::nn::Module{

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
      out[i] = 0;
    }else{
      out[i] = 1;
    }
  }
  return out;
}

void train(Net& model, const torch::Tensor& train_input, 
          torch::Tensor train_output, unsigned int batch_size=10){
  
  // Specify the loss function
  auto loss = torch::mse_loss;
  
  // Define the number of epochs to train the network
  unsigned int epochs = 25;

  // Set the learning rate
  double eta = 0.1;

  // Define optimizer
  torch::optim::SGD optimizer(model.parameters(),
    torch::optim::SGDOptions(eta).momentum(0.0)); 
  
  double sum_loss = 0.;
  for(uint e=0; e<epochs; e++){
    sum_loss = 0.;
    for(uint b=0; b<train_input.size(0); b+=batch_size){
      torch::Tensor output = model.forward(
        torch::narrow(train_input,0,b,batch_size));
      torch::Tensor loss = torch::mse_loss(output.view(-1),
        train_output.narrow(0,b,batch_size));
      
      sum_loss += loss.item().toDouble();
      model.zero_grad();
      loss.backward();

      optimizer.step();
    }
    std::cout << "Sum of loss at epoch " << e 
      << ": " << sum_loss << std::endl;
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

#elif defined(LEARN) 
  Net model = Net(); 

  torch::Tensor data  = create2DDataSet(DIM);
  torch::Tensor truth = createGroundTruth(data);

  train(model, data, truth);
  
#endif
  return EXIT_SUCCESS;
}
