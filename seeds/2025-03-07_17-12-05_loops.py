import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for input_tensor in inputs:
            sum_result = input_tensor.sum()
            mean_result = input_tensor.mean()
            max_result = input_tensor.max()
            min_result = input_tensor.min()
            results.append((sum_result, mean_result, max_result, min_result))
        return results

input_tensors = [
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 2D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 3D tensor
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]),  # 4D tensor
    torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32),  # 2D tensor
    torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)  # 1D tensor
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors)
    print(outputs)