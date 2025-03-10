import torch

class PtModule(torch.nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            result = torch.sum(tensor)
            results.append(result)
        return results
input_tensors = [torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32), torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32), torch.tensor([[[1, 0], [0, 1]], [[1, 2], [2, 1]]], dtype=torch.float32), torch.tensor([[2, 2], [2, 2]], dtype=torch.float32), torch.tensor([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    output = model(input_tensors)
    print(output)