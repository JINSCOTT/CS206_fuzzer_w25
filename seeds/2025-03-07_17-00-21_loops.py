import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            # Adding a constant value
            added = tensor + 2
            # Multiplying by a constant value
            multiplied = tensor * 3
            # Subtracting a constant value
            subtracted = tensor - 1
            # Dividing by a constant value
            divided = tensor / 2
            
            results.append((added, multiplied, subtracted, divided))
        
        return results

# Define input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # 3D tensor
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),  # 3D tensor
    torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]]),  # 4D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 4D tensor
    torch.tensor([[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]])  # 4D tensor
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors)
    for i, output in enumerate(outputs):
        print(f"Output for tensor {i}:")
        for operation in output:
            print(operation)