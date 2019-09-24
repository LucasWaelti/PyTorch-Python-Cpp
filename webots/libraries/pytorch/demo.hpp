#include <iostream>
#include <math.h>
#include <torch/torch.h>
#include <torch/script.h>

//#define BASICS 
#define LEARN
#define DIM 100 // Dataset size
#define RADIUS 0.5
#define BATCH_SIZE 10

struct NetImpl: torch::nn::Module{

    NetImpl();
    
    torch::Tensor forward(torch::Tensor x);

    torch::nn::Linear _in{nullptr},_h{nullptr},_out{nullptr};
};

// create a module holder, which is a std::shared_ptr<NetImpl>.
// usage: 
//  Net model = Net();
//  model->forward(); (instead of model.forward())
TORCH_MODULE(Net);

torch::Tensor create2DDataSet(unsigned int dim=DIM);

double rule(double x, double y, double r=RADIUS); 

torch::Tensor createGroundTruth(const torch::Tensor& input);

void train(Net& model, const torch::Tensor& train_input, 
          torch::Tensor train_output, 
          unsigned int batch_size=BATCH_SIZE);

void evalAccuracy(Net& model, torch::Tensor test_input, 
                  torch::Tensor test_output);

void evalAccuracy(torch::Tensor test_output, torch::Tensor output);

void save(Net& model, std::string path);

void load(Net& model, std::string path);

torch::jit::script::Module deserializeModel(std::string path);

void demo(bool training=true);