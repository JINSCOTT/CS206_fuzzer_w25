import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()
    
    def forward(self, input_tensor):
        # Perform some mathematical operations
        output_tensor = input_tensor * 2  # Example operation: scaling
        for i in range(output_tensor.size(0)):
            output_tensor[i] = output_tensor[i] + 1  # Adding 1 to each element in batch
        return output_tensor

# Predefined input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),  # 4D tensor
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 2D tensor (will be treated as batch)
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]]),  # 3D tensor
    torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])  # 2D tensor
]

if __name__ == "__main__":
    pt_module = PtModule()
    for input_tensor in input_tensors:
        output_tensor = pt_module(input_tensor)
        print("Output Tensor:", output_tensor)