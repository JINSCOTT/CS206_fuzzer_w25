import torch

class PtModule(torch.nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        output = input_tensor.clone()
        for i in range(output.size(0)):
            output[i] = torch.add(torch.mul(output[i], 2), 1)
        return output
input_tensors = [torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32), torch.tensor([[[13, 14], [15, 16], [17, 18]]], dtype=torch.float32), torch.tensor([[[19, 20, 21, 22]], [[23, 24, 25, 26]]], dtype=torch.float32), torch.tensor([[[27], [28], [29]], [[30], [31], [32]]], dtype=torch.float32), torch.tensor([[[33, 34, 35], [36, 37, 38]], [[39, 40, 41], [42, 43, 44]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        output = model(tensor)
        print(output)