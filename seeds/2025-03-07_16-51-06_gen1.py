import torch

class PtModule(torch.nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        output_tensor = input_tensor.clone()
        for i in range(output_tensor.size(0)):
            output_tensor[i] = output_tensor[i] + 2 * (i + 1)
        return output_tensor

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),      # 3D tensor
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]),  # 4D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]),  # 3D tensor
    torch.tensor([[[[1]], [[2]]], [[[3]], [[4]]], [[[5]], [[6]]]])              # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print("Input:\n", input_tensor)
        print("Output:\n", output)
        print()  # Print a newline for better readability