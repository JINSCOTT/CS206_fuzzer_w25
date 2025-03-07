import torch

class PtModule(torch.nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        output = input_tensor.clone()
        for i in range(output.shape[0]):
            output[i] = torch.add(torch.mul(output[i], 2), 1)
        return output
input_tensors = [torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), torch.tensor([[1, 2], [3, 4], [5, 6]]), torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1], [2]], [[3], [4]]]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])]
if __name__ == '__main__':
    pt_module = PtModule()
    for tensor in input_tensors:
        result = pt_module(tensor)
        print(result)