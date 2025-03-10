import torch

class PtModule:

    def __init__(self):
        pass

    def forward(self, tensors):
        results = []
        for tensor in tensors:
            result = torch.add(torch.mean(tensor), torch.std(tensor))
            results.append(result)
        return results
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=torch.float32), torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32), torch.tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=torch.float32)]
if __name__ == '__main__':
    pt_module = PtModule()
    results = pt_module.forward(input_tensors)
    print(results)