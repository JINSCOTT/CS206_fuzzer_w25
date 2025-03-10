import torch

class PtModule:

    def __init__(self):
        pass

    def forward(self, input_tensor):
        result = input_tensor.clone()
        for i in range(input_tensor.size(0)):
            result[i] = torch.mul(input_tensor[i], 2)
        return result
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]), torch.tensor([[10, 20, 30], [40, 50, 60]]), torch.tensor([1, 2, 3, 4]), torch.tensor([[[[1]], [[2]]], [[[3]], [[4]]]])]
if __name__ == '__main__':
    module = PtModule()
    for tensor in input_tensors:
        output = module.forward(tensor)
        print(f'Input: {tensor}\nOutput: {output}\n')