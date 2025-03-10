import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        output = []
        for i in range(input_tensor.size(0)):
            # Simple operation: multiply by 2 and add 3
            result = input_tensor[i] * 2 + 3
            output.append(result)
        return torch.stack(output)

# Define input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
    torch.tensor([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]),
    torch.tensor([[[17.0, 18.0], [19.0, 20.0]], [[21.0, 22.0], [23.0, 24.0]]]),
    torch.tensor([[[25.0, 26.0], [27.0, 28.0]], [[29.0, 30.0], [31.0, 32.0]]]),
    torch.tensor([[[33.0, 34.0], [35.0, 36.0]], [[37.0, 38.0], [39.0, 40.0]]])
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output_tensor = model(input_tensor)
        print(output_tensor)