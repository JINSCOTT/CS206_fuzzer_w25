import torch

class PtModule(torch.nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        results = []
        for i in range(x.size(0)):
            result = torch.sum(x[i]) * i
            results.append(result)
        return torch.stack(results)

input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], dtype=torch.float32),
    torch.tensor([[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]], dtype=torch.float32),
    torch.tensor([[[5, 5], [5, 5]], [[10, 10], [10, 10]], [[15, 15], [15, 15]]], dtype=torch.float32),
    torch.tensor([[[6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17]]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(output)