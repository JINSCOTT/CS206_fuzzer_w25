import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            tensor_mean = torch.mean(tensor)  # Example operation
            tensor_sum = torch.sum(tensor)    # Example operation
            results.append((tensor_mean, tensor_sum))
        return results

# Define input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),          # 3D tensor
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]),  # 4D tensor
    torch.tensor([[1.0, 2.0, 3.0]]),                   # 2D tensor
    torch.tensor([[1.0], [2.0], [3.0]]),               # 2D tensor
    torch.tensor([[[1.0]]])                            # 3D tensor
]

if __name__ == "__main__":
    model = PtModule()
    output = model(input_tensors)
    print(output)