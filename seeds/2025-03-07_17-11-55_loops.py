import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            sum_tensor = torch.sum(tensor)
            mean_tensor = torch.mean(tensor)
            product_tensor = torch.prod(tensor)
            results.append((sum_tensor, mean_tensor, product_tensor))
        return results

input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32),
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32),
    torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.float32),
    torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors)
    print(outputs)