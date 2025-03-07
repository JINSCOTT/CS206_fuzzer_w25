import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for inp in inputs:
            # Performing some math operations
            output = inp + 2  # Add 2 to each element
            output = output * 3  # Multiply by 3
            output = output - 1  # Subtract 1
            results.append(output)  # Store the result
        return results

# Define the input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
    torch.tensor([[[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]]),
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    torch.tensor([[[2.0]], [[3.0]], [[4.0]]]),
    torch.tensor([[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]])
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors)
    for i, output in enumerate(outputs):
        print(f"Output tensor {i}: {output}")