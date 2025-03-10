import torch

class PtModule:

    def __init__(self):
        pass

    def forward(self, input_tensor):
        output_tensor = torch.mul(input_tensor, 2)
        for i in range(output_tensor.size(0)):
            output_tensor[i] = torch.add(output_tensor[i], 1)
        return output_tensor
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[9, 10, 11], [12, 13, 14]], [[15, 16, 17], [18, 19, 20]]]), torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[[-1, -2], [-3, -4]]]), torch.tensor([[[[1], [2]], [[3], [4]]]])]
if __name__ == '__main__':
    module = PtModule()
    for tensor in input_tensors:
        output = module.forward(tensor)
        print(f'Input Tensor:\n{tensor}\nOutput Tensor:\n{output}\n')