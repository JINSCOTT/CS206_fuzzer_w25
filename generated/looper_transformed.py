import torch

class PtModule(torch.nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = x.clone()
        for i in range(x.shape[0]):
            result[i] = torch.add(torch.mul(result[i], 2), 3)
        return result
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]]), torch.tensor([[[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]]]), torch.tensor([[[[15.0], [16.0]], [[17.0], [18.0]]]]), torch.tensor([[[[[19.0], [20.0]]]]])]

def main():
    module = PtModule()
    for tensor in input_tensors:
        output = module(tensor)
        print('Input Tensor:\n', tensor)
        print('Output Tensor:\n', output)
if __name__ == '__main__':
    main()