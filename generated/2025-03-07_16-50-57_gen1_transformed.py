import torch

class PtModule:

    def __init__(self):
        pass

    def forward(self, input_tensor):
        result = input_tensor.clone()
        for i in range(result.shape[0]):
            result[i] = torch.mul(result[i], 2)
        return result
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[9, 10, 11], [12, 13, 14]], [[15, 16, 17], [18, 19, 20]]]), torch.tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]), torch.tensor([[[1, 2], [3, 4]]]), torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]])]
if __name__ == '__main__':
    pt_module = PtModule()
    for input_tensor in input_tensors:
        output = pt_module.forward(input_tensor)
        print(f'Input: \n{input_tensor}\nOutput: \n{output}\n')