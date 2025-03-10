import torch

class PtModule(torch.nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = []
        for i in range(x.shape[0]):
            tensor_sum = torch.add(x[i], torch.inverse(x[i]))
            result.append(tensor_sum)
        return result
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32), torch.tensor([[[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32), torch.tensor([[[9.0, 10.0], [11.0, 12.0]]], dtype=torch.float32), torch.tensor([[[13.0, 14.0], [15.0, 16.0]]], dtype=torch.float32), torch.tensor([[[17.0, 18.0], [19.0, 20.0]]], dtype=torch.float32)]
if __name__ == '__main__':
    pt_module = PtModule()
    for tensor in input_tensors:
        output = pt_module(tensor)
        print(output)