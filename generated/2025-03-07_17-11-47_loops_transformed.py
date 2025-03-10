import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        for i in range(x.shape[0]):
            x[i] = torch.add(x[i], 2)
            x[i] = torch.mul(x[i], 3)
            x[i] = torch.sub(x[i], 5)
            x[i] = torch.div(x[i], 2)
        return x
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[5.0], [6.0]], [[7.0], [8.0]]]), torch.tensor([[[9.0, 10.0, 11.0]]]), torch.tensor([[[12.0, 13.0]], [[14.0, 15.0]], [[16.0, 17.0]]]), torch.tensor([[[18.0]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output_tensor = model(input_tensor)
        print(f'Input:\n{input_tensor}\nOutput:\n{output_tensor}\n')