import torch

class PtModule:

    def __init__(self):
        pass

    def forward(self, input_tensor):
        result = input_tensor.clone()
        for i in range(input_tensor.size(0)):
            result[i] = torch.add(torch.mul(result[i], 2), 1)
        return result
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]]), torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]]), torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]]), torch.tensor([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]])]
if __name__ == '__main__':
    module = PtModule()
    for tensor in input_tensors:
        output = module.forward(tensor)
        print(output)