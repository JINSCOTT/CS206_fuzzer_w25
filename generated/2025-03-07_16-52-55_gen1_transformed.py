import torch

class PtModule:

    def __init__(self):
        pass

    def forward(self, inputs):
        results = []
        for i in range(len(inputs)):
            result = torch.add(torch.mul(inputs[i], 2), 3)
            results.append(result)
        return results
input_tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]), torch.tensor([[5, 10, 15], [20, 25, 30]]), torch.tensor([[[10, 20], [30, 40]], [[50, 60], [70, 80]], [[90, 100], [110, 120]]])]
if __name__ == '__main__':
    pt_module = PtModule()
    output = pt_module.forward(input_tensors)
    print(output)