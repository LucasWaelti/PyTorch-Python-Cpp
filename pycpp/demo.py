import math 
import torch 
from torch.nn import functional as F

BASICS = False
LEARN = True 
DIM = 10 # Dataset size


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
        x = torch.sigmoid(self._out(x))

        # return the output
        return x 





def create2DDataSet(dim:int):
    return torch.rand([dim,2])


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
    
    # Specify the loss function
    loss = torch.nn.MSELoss() 
    
    # Define the number of epochs to train the network
    epochs = 25
    
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

        data  = create2DDataSet(DIM)
        truth = createGroundTruth(data)

        model = train(model, data, truth) 












if __name__ == "__main__":
    main() 