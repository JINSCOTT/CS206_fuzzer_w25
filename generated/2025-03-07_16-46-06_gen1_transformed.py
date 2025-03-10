import torch

class PtModule:

    def __init__(self):
        pass

    def forward(self, input_tensor):
        result = torch.mul(input_tensor, 2)
        for i in range(3):
            result = torch.add(result, input_tensor)
        return result
input_tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]), torch.tensor([[7, 8, 9], [10, 11, 12]]), torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]])]
if __name__ == '__main__':
    pt_module = PtModule()
    for tensor in input_tensors:
        output = pt_module.forward(tensor)
        print(f'Input:\n{tensor}\nOutput:\n{output}\n')