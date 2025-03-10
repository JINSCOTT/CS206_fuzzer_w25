import torch

class PtModule(torch.nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        result = input_tensor.clone()
        for i in range(input_tensor.size(0)):
            result[i] = input_tensor[i] * 2 + 1
        return result

input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1]], [[2]]], [[[3]], [[4]]]], dtype=torch.float32),  # 4D tensor
    torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=torch.float32),  # 2D tensor
    torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]], [[[5.0], [6.0]], [[7.0], [8.0]]]], dtype=torch.float32)  # 4D tensor
]

if __name__ == "__main__":
    module = PtModule()
    for tensor in input_tensors:
        output = module(tensor)
        print(f"Input:\n{tensor}\nOutput:\n{output}\n")