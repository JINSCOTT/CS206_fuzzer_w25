import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Perform some mathematical operations
        result = input_tensor * 2  # Example operation: scaling by 2
        for i in range(result.size(0)):
            result[i] = result[i] + 1  # Example operation: adding 1 to each element
        return result

# Defining the input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]),  # 4D tensor
    torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),  # 3D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]),  # 4D tensor
    torch.tensor([[5.0, 15.0], [25.0, 35.0], [45.0, 55.0]])  # 3D tensor
]

if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f"Output for input tensor:\n{tensor}\n is:\n{output}\n")