import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        # Perform various mathematical operations
        x = x * 2.5                      # Scalar multiplication
        x = x - torch.mean(x)            # Subtract mean
        x = torch.exp(x)                 # Exponential
        x = x / (torch.abs(x) + 1)       # Normalize with abs
        x = torch.matmul(x, x.transpose(-1, -2)) if x.dim() == 3 else x  # Matmul if 3D
        x = x **0.5
        return x

# Array of potential input tensors (3D and 4D)
input_tensors = [
    torch.tensor([[[ 0.5, -1.2,  0.3],
                   [ 1.5,  0.2, -0.8],
                   [-0.5,  0.9,  1.1]],

                  [[-0.3,  0.8, -1.5],
                   [ 0.6, -0.7,  0.4],
                   [ 1.2,  0.1, -0.9]],

                  [[ 0.7, -0.6,  0.5],
                   [-1.1,  0.3,  0.2],
                   [ 0.4, -0.2,  0.9]]]),  # 3D tensor

    torch.tensor([[[[ 0.2, -0.5,  0.6, -0.1],
                    [ 0.7,  0.3, -0.8,  0.9],
                    [-0.4,  1.0, -0.2,  0.5],
                    [ 0.6, -0.7,  0.8, -0.3]],

                   [[-0.6,  0.7, -0.9,  0.4],
                    [ 0.5, -0.2,  0.3, -0.8],
                    [ 0.9,  0.1, -0.5,  0.6],
                    [-0.3,  0.8, -0.7,  0.2]]]]),  # 4D tensor
]

if __name__ == "__main__":
    # Simple check to verify the module runs
    model = PtModule()
    sample_input = input_tensors[0]  # Pick the first tensor
    output = model(sample_input)
    print("Sample Input Shape:", sample_input.shape)
    print("Output Shape:", output.shape)
    print("Output Tensor:\n", output)
