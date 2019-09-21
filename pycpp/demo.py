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
        x = F.sigmoid(self._out(x))

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
        print(data)
        print(truth) 













if __name__ == "__main__":
    main() 