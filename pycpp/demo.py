
import torch 

BASICS = True
DIM = 4 # Dataset size

def create2DDataSet(dim:int):
    return torch.rand([dim,2])


def createGroundTruth(input:torch.tensor):
    out = torch.empty(input.size(0))
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
















if __name__ == "__main__":
    main() 