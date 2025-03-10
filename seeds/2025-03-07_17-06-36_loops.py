import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Perform some basic math operations
        output_tensor = input_tensor + 2  # Addition
        output_tensor = output_tensor * 3  # Multiplication
        
        # Apply a loop to process each element
        for i in range(output_tensor.size(0)):
            output_tensor[i] = output_tensor[i].pow(2)  # Squaring each element
        
        return output_tensor

# Define input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
    torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
    torch.tensor([[[2.0], [3.0]], [[4.0], [5.0]], [[6.0], [7.0]]]),
    torch.tensor([[[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]], [[4.0, 4.0]]]),
    torch.tensor([[[1.0, 2.0, 3.0]]])
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    model = PtModule()
    for i, input_tensor in enumerate(input_tensors):
        output = model(input_tensor)
        print(f"Output for input tensor {i}: \n{output}")