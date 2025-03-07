import torch

class PtModule(torch.nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        output = input_tensor.clone()
        for i in range(output.shape[0]):
            output[i] = output[i] * 2 + 1  # Example operation: double and add one
        return output

# Define input tensors with specific values
input_tensors = [
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),  # 3D tensor
    torch.tensor([[1, 2], [3, 4], [5, 6]]),  # 2D tensor
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D tensor with floats
    torch.tensor([[[1], [2]], [[3], [4]]]),  # 3D tensor with single element in last dimension
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])  # 4D tensor
]

if __name__ == "__main__":
    pt_module = PtModule()
    for tensor in input_tensors:
        result = pt_module(tensor)
        print(result)