import torch

class PtModule:

    def __init__(self):
        pass

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            result = torch.mul(torch.sum(tensor), 0.1)
            results.append(result)
        return results
input_tensors = [torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]]), torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]]), torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]])]
if __name__ == '__main__':
    pt_module = PtModule()
    outputs = pt_module.forward(input_tensors)
    for (i, output) in enumerate(outputs):
        print(f'Output for input tensor {torch.add(i, 1)}: {output.item()}')