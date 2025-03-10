import torch

class PtModule(torch.nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result = input_tensor.clone()
        for i in range(result.size(0)):
            result[i] = torch.add(torch.mul(result[i], 2), 1)
        return result
input_tensors = [torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), torch.tensor([[[1, 0], [3, 0]], [[0, 2], [0, 4]]]), torch.tensor([[[10, 20, 30], [40, 50, 60]]]), torch.tensor([[[1, 2, 3]], [[4, 5, 6]]]), torch.tensor([[[9], [8], [7]], [[6], [5], [4]], [[3], [2], [1]]])]
if __name__ == '__main__':
    pt_module = PtModule()
    for input_tensor in input_tensors:
        output = pt_module(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output}\n')