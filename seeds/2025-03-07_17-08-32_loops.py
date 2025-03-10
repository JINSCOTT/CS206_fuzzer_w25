import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Perform various mathematical operations
        output = input_tensor + 2  # Add 2 to all elements
        output = output * 3        # Multiply all elements by 3

        # Loop through each element in the tensor
        for i in range(output.size(0)):
            for j in range(output.size(1)):
                for k in range(output.size(2)):
                    output[i, j, k] = output[i, j, k] ** 2  # Square each element

        return output


# Define input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
    torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),
    torch.tensor([[[9.0, 10.0], [11.0, 12.0]]]),
    torch.tensor([[[13.0, 14.0], [15.0, 16.0]]]),
    torch.tensor([[[17.0, 18.0], [19.0, 20.0]]])
]

# Main section to check if the script runs
if __name__ == "__main__":
    model = PtModule()
    for i, input_tensor in enumerate(input_tensors):
        output_tensor = model(input_tensor)
        print(f"Output for input tensor {i+1}:\n{output_tensor}")