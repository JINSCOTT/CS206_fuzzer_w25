import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        output = input_tensor.clone()
        for i in range(output.shape[0]):
            output[i] += i
        return output
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]), torch.tensor([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]), torch.tensor([[[[17.0, 18.0], [19.0, 20.0]], [[21.0, 22.0], [23.0, 24.0]]]]), torch.tensor([[[25.0, 26.0], [27.0, 28.0]], [[29.0, 30.0], [31.0, 32.0]]])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(f'Input Tensor:\n{tensor}\nOutput Tensor:\n{output}\n')