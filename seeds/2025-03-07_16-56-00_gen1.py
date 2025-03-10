import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = []
        for tensor in x:
            # Perform some simple math operations
            mean = torch.mean(tensor)
            std = torch.std(tensor)
            result.append((mean, std))
        return result

# Defining input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),                    # 2D tensor
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]]),      # 3D tensor
    torch.tensor([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]]),                  # 2D tensor
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]])  # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors)
    print(outputs)