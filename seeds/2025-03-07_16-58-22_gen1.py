import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Perform some mathematical operations
        output_tensor = input_tensor * 2 + 5
        for i in range(output_tensor.size(0)):
            output_tensor[i] = output_tensor[i].sum()  # sum each tensor in the batch
        return output_tensor

# Define input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
    torch.tensor([[[2.0, 3.0], [4.0, 5.0]], [[6.0, 7.0], [8.0, 9.0]]]),
    torch.tensor([[[3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0]]]),
    torch.tensor([[[4.0, 5.0], [6.0, 7.0]], [[8.0, 9.0], [10.0, 11.0]]]),
    torch.tensor([[[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]])
]

if __name__ == "__main__":
    pt_module = PtModule()
    for tensor in input_tensors:
        output = pt_module(tensor)
        print(f"Output tensor: {output}")