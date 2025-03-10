import torch

class PtModule:

    def __init__(self):
        pass

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            result = torch.sum(torch.pow(tensor, 2))
            results.append(result)
        return results
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[5.0], [10.0], [15.0]]), torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]])]
if __name__ == '__main__':
    pt_module = PtModule()
    outputs = pt_module.forward(input_tensors)
    print(outputs)