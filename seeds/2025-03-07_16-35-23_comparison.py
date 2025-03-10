import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Example operations
        added = x + 2
        multiplied = x * 3
        compared = x > 1
        subtracted = x - 4
        divided = x / 2

        return added, multiplied, compared, subtracted, divided

# Input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]]),  # 2D tensor
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]),  # 4D tensor
    torch.tensor([[1, -1, 2]], dtype=torch.float32),  # 2D tensor with float
    torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int32)  # 2D tensor with int
]

def main():
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input Tensor:\n{input_tensor}\nOutputs:\n{output}\n")

if __name__ == "__main__":
    main()