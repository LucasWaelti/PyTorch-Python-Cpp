#include "demo.hpp"

torch::Tensor create2DDataSet(unsigned int dim){
  return 2 * torch::rand({dim,2}) - 1;
}
void printHello(){
  std::cout << "Hello from archive!! - new version!!" << std::endl;
}

NetImpl::NetImpl(){
  // Constructor - build the network's layers
  this->_in = register_module("in",torch::nn::Linear(2,10));
  this->_h = register_module("h",torch::nn::Linear(10,5));
  this->_out = register_module("out",torch::nn::Linear(5,1));
}

torch::Tensor NetImpl::forward(torch::Tensor x){
  // apply ReLU activations and sigmoid for the output
  x = torch::relu(this->_in->forward(x));
  x = torch::relu(this->_h->forward(x));
  x = torch::sigmoid(this->_out->forward(x));
    
  // return the output
  return x;
}



double rule(double x, double y, double r){
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
          torch::Tensor train_output, unsigned int batch_size){
  // Set the model in training state. Otherwise .eval()
  model->train();

  // Specify the loss function
  // Use MSE loss -> see loops below
  
  // Define the number of epochs to train the network
  unsigned int epochs = 100;

  // Set the learning rate
  double eta = 0.1;

  // Define optimizer
  torch::optim::SGD optimizer(model->parameters(),
    torch::optim::SGDOptions(eta).momentum(0.0)); 
  
  
  double sum_loss = 0.;
  for(uint e=0; e<epochs; e++){
    sum_loss = 0.;
    for(uint b=0; b<train_input.size(0); b+=batch_size){
      torch::Tensor output = model->forward(
        torch::narrow(train_input,0,b,batch_size));
      torch::Tensor loss = torch::mse_loss(output.view(-1),
        train_output.narrow(0,b,batch_size));
      
      sum_loss += loss.item().toDouble();
      model->zero_grad();
      loss.backward();

      optimizer.step();
    }
    std::cout << "Sum of loss at epoch " << e 
      << ": " << sum_loss << "\t\r" << std::flush;
  }
  std::cout << std::endl;
}

void evalAccuracy(Net& model, torch::Tensor test_input, 
                  torch::Tensor test_output){
  model->eval();

  torch::Tensor output = model->forward(test_input).view(-1);
  for(uint i=0; i<output.size(0); i++){
    output[i] = output[i].item().toDouble()<0.5 ? 0 : 1;
  }
  output = test_output == output;

  uint count = 0;
  for(uint r=0; r<output.size(0); r++){
    if(!output[r].item().toBool()){
      count++;
    }
  }
  double accuracy = (1.-(double)count/(double)output.size(0))*100.;
  std::cout << "Accuracy: " << accuracy << "%" << std::endl; 
}
void evalAccuracy(torch::Tensor test_output, torch::Tensor output){
  for(uint i=0; i<output.size(0); i++){
    output[i] = output[i].item().toDouble()<0.5 ? 0 : 1;
  }
  output = test_output == output;

  uint count = 0;
  for(uint r=0; r<output.size(0); r++){
    if(!output[r].item().toBool()){
      count++;
    }
  }
  double accuracy = (1.-(double)count/(double)output.size(0))*100.;
  std::cout << "Accuracy: " << accuracy << "%" << std::endl; 
}

void save(Net& model, std::string path){
  // Save the model
  torch::save(model,path); 
}

void load(Net& model, std::string path){
  // Load the model
  torch::load(model,path); 
}

torch::jit::script::Module deserializeModel(std::string path){
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(path); 
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
  }
  return module;
}

void demo(bool training){
  Net model = Net(); 
  torch::jit::script::Module traced,scripted;

  if(training){
    torch::Tensor data  = create2DDataSet(DIM);
    torch::Tensor truth = createGroundTruth(data);
    train(model, data, truth);
    save(model,"./models/model_cpp.pt");
  } 
  else{ 
    load(model,"./models/model_cpp.pt"); 
    traced = deserializeModel("./models/traced.pt");
    scripted = deserializeModel("./models/scripted.pt");
  }

  torch::Tensor test_input  = create2DDataSet(DIM);
  torch::Tensor test_output = createGroundTruth(test_input);
  evalAccuracy(model,test_input,test_output);

  if(!training){
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(test_input);

    // Execute the model and turn its output into a tensor.
    torch::Tensor out_traced = traced.forward(inputs).toTensor().view(-1);
    torch::Tensor out_scripted = scripted.forward(inputs).toTensor().view(-1);
    std::cout << out_traced << std::endl;
    evalAccuracy(test_output,out_traced);
    evalAccuracy(test_output,out_scripted);
  }
}
