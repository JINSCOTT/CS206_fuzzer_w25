import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        output_tensor = torch.sub(torch.mul(torch.add(input_tensor, 2), 3), 4)
        for i in range(output_tensor.shape[0]):
            for j in range(output_tensor.shape[1]):
                for k in range(output_tensor.shape[2]):
                    output_tensor[i, j, k] = torch.relu(output_tensor[i, j, k])
        return output_tensor
input_tensors = [torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32), torch.tensor([[[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=torch.float32), torch.tensor([[[-1, -2, -3], [-4, -5, -6]]], dtype=torch.float32), torch.tensor([[[1]], [[2]], [[3]], [[4]]], dtype=torch.float32), torch.tensor([[[[1, 2]], [[3, 4]]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output_tensor = model(input_tensor)
        print(f'Input Tensor:\n{input_tensor}\nOutput Tensor:\n{output_tensor}\n')