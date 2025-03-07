import torch

class PtModule:

    def __init__(self):
        pass

    def forward(self, input_tensors):
        results = []
        for tensor in input_tensors:
            result = torch.add(torch.sum(tensor), torch.mean(tensor))
            results.append(result)
        return results
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]]), torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]), torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]), torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]])]
if __name__ == '__main__':
    module = PtModule()
    outputs = module.forward(input_tensors)
    print(outputs)