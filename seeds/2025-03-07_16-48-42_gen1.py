import torch

class PtModule(torch.nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        output_tensor = input_tensor.clone()
        for i in range(output_tensor.size(0)):
            output_tensor[i] = output_tensor[i] * 2  # Simple operation: multiply by 2
        return output_tensor

# Input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
    torch.tensor([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]),
    torch.tensor([[[10, 20], [30, 40]], [[50, 60], [70, 80]], [[90, 100], [110, 120]]]),
    torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
]

# Main section to test the script
if __name__ == "__main__":
    pt_module = PtModule()
    for input_tensor in input_tensors:
        output = pt_module(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")