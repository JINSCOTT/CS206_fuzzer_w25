import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        # Applying some mathematical operations
        added = input_tensor + 2
        multiplied = input_tensor * 3
        divided = input_tensor / 2
        
        # Loop through the input tensor
        result = []
        for i in range(input_tensor.size(0)):
            result.append(added[i] + multiplied[i] - divided[i])
        
        return torch.stack(result)

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=torch.float32),
    torch.tensor([[[17, 18], [19, 20]]], dtype=torch.float32),
    torch.tensor([[[21, 22, 23]], [[24, 25, 26]], [[27, 28, 29]]], dtype=torch.float32),
    torch.tensor([[[30]], [[31]], [[32]]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f"Input:\n{input_tensor}\nOutput:\n{output}\n")