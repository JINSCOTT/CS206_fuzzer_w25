import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations; feel free to modify as needed
        output = []
        for tensor in x:
            # Adding 1 to each element in the tensor
            added = tensor + 1
            # Multiplying each element by 2
            multiplied = added * 2
            # Collect results
            output.append(multiplied)
        return output

# Input tensors with explicitly defined values
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]),                       # 2D tensor
    torch.tensor([[[1], [2]], [[3], [4]]]),                   # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]),    # 4D tensor
    torch.tensor([[7, 8], [9, 10]]),                           # 2D tensor
    torch.tensor([[[2], [4]], [[6], [8]]])                    # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    output = model(input_tensors)
    for i, out in enumerate(output):
        print(f"Output for input tensor {i+1}:\n{out}")