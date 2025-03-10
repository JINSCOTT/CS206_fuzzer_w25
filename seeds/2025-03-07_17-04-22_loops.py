import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        result = []

        for tensor in inputs:
            # Example of some math operations
            sum_tensor = torch.sum(tensor)
            mean_tensor = torch.mean(tensor)
            max_tensor = torch.max(tensor)
            min_tensor = torch.min(tensor)

            result.append({
                'sum': sum_tensor,
                'mean': mean_tensor,
                'max': max_tensor,
                'min': min_tensor
            })

        return result

input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]),        # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 2D tensor with floats
    torch.tensor([[[[1], [2]], [[3], [4]]]]),            # 4D tensor
    torch.tensor([5, 10, 15, 20])                       # 1D tensor
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors)
    print(outputs)