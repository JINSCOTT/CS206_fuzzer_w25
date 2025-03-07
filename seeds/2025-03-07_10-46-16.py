import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        result = []
        for i in range(x.shape[0]):
            temp = x[i] * 2  # Multiplying each element by 2
            temp = temp + 3  # Adding 3 to each element
            temp = temp / 2  # Dividing each element by 2
            result.append(temp)
        return torch.stack(result)

# Sample input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]),
    torch.tensor([[7, 8, 9], [10, 11, 12]]),
    torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]),
    torch.tensor([[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]])
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")