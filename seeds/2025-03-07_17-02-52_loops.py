import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = {}
        
        # Adding a constant to the input
        result['add'] = x + 2
        
        # Subtracting a constant from the input
        result['subtract'] = x - 1
        
        # Multiplying the input by a constant
        result['multiply'] = x * 3
        
        # Dividing the input by a constant
        result['divide'] = x / 4
        
        # Looping through the input tensor
        loop_result = []
        for i in range(x.shape[0]):
            loop_result.append(x[i] * i)
        result['loop'] = torch.stack(loop_result)
        
        return result

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),     # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),   # 3D tensor
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]), # 4D tensor
    torch.tensor([[5.0, 6.0], [7.0, 8.0]]),     # 2D tensor
    torch.tensor([[[9.0, 10.0], [11.0, 12.0]]]) # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for x in input_tensors:
        output = model(x)
        print(output)