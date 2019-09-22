import math 
import torch 
from torch.nn import functional as F

BASICS = False
LEARN = True 
DIM = 100 # Dataset size


class Net(torch.nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self._in = torch.nn.Linear(2, 10)
        self._h = torch.nn.Linear(10, 5)
        self._out = torch.nn.Linear(5, 1)

    
    def forward(self, x):
        # apply ReLU activations and sigmoid for the output
        x = F.relu(self._in(x))
        x = F.relu(self._h(x))
        x = torch.tanh(self._out(x))

        # return the output
        return x 










def create2DDataSet(dim:int):
    return 2*torch.rand([dim,2])-1


def rule(x,y,r=0.5):
    # 0 if contained in circle with radius r, 
    # 1 if outside
    if math.sqrt(x**2 + y**2) < r:
        return 0
    else:
        return 1 



def createGroundTruth(inp:torch.tensor):
    out = torch.empty(inp.size(0))
    for i,point in enumerate(inp):
        if rule(point[0],point[1]) == 0:
            out[i] = 0
        else:
            out[i] = 1
    return out 





def train(model:Net, train_input:torch.Tensor, 
    train_output:torch.Tensor, batch_size=10):
    # Set the model in training state. Otherwise .eval()
    model.train()
    
    # Specify the loss function
    loss = torch.nn.MSELoss() 
    
    # Define the number of epochs to train the network
    epochs = 100
    
    # Set the learning rate
    eta = 0.1

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=eta, momentum=0.0)

    
    
    
    
    
    
    
    for e in range(epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), batch_size):
            # Narrow: # tensor,dim,start,length
            output = model(train_input.narrow(0, b, batch_size)) 
            l = loss(output.view(-1), 
                    train_output.narrow(0,b,batch_size).float())
             
            sum_loss = sum_loss + l.item()
            model.zero_grad()
            l.backward()

            optimizer.step()

        print('Sum of loss at epoch {}: \t'.format(e),sum_loss,end='\r')
    print('')
    return model



def evalAccuracy(model:Net, test_input:torch.Tensor, 
    test_output:torch.Tensor):
    model.eval()

    output = model(test_input).view(-1)
    for i,v in enumerate(output):
        output[i] = 0 if v < 0.5 else 1 
    result = test_output == output
     
    count = 0
    for r in result:  
        if r == False:
            count += 1
    print("Accuracy: ",(1-count/len(result))*100,"%")






def save(model:Net,path:str):
    # Save the model 
    torch.save(model,path)


def load(path:str):
    # Load the model 
    return torch.load(path) 


def main():
    print("Running Python implementation\n")
    
    if BASICS:
        print("Basic Tensor manipulations:\n")

        print("Manually initialized tensor:")
        t = torch.tensor([[1,2],[3,4]])

        print(t) 

        print("Transpose a tensor:")
        t = torch.transpose(t,0,1)
        print(t,'\n')

        print("Concatenate tensors:")
        t1 = torch.tensor([[5,6]])
        
        t = torch.cat([t,t1],0)
        print(t)

    elif LEARN:
        model = Net()

        training = False
        if training is True:
            data  = create2DDataSet(DIM)
            truth = createGroundTruth(data)
            model = train(model, data, truth) 
            save(model,"./models/model_py.pt")
        else:
            model = load("./models/model_py.pt")

        test_input  = create2DDataSet(DIM)
        test_output = createGroundTruth(test_input)
        evalAccuracy(model, test_input, test_output)

if __name__ == "__main__":
    main() 

