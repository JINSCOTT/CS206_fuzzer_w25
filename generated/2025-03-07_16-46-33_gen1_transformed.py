import torch

class PtModule(torch.nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result = torch.zeros_like(input_tensor)
        for i in range(input_tensor.size(0)):
            result[i] = torch.add(torch.mul(input_tensor[i], 2), 3)
        return result
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[9, 10, 11], [12, 13, 14]], [[15, 16, 17], [18, 19, 20]]]), torch.tensor([[[21, 22], [23, 24]], [[25, 26], [27, 28]], [[29, 30], [31, 32]]]), torch.tensor([[[33], [34]], [[35], [36]], [[37], [38]], [[39], [40]]]), torch.tensor([[[41, 42, 43]], [[44, 45, 46]], [[47, 48, 49]], [[50, 51, 52]]])]
if __name__ == '__main__':
    pt_module = PtModule()
    for tensor in input_tensors:
        output = pt_module(tensor)
        print(f'Input Tensor:\n{tensor}\nOutput Tensor:\n{output}\n')