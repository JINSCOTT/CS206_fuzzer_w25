import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        output = torch.mul(input_tensor, 2)
        for i in range(output.shape[0]):
            output[i] = torch.add(output[i], i)
        return output
input_tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]), torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]]])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output_tensor = model(tensor)
        print(f'Input Tensor:\n{tensor}\nOutput Tensor:\n{output_tensor}\n')