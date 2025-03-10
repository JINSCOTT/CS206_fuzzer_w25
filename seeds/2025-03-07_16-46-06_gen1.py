import torch

class PtModule:
    def __init__(self):
        pass

    def forward(self, input_tensor):
        # Simple operations for demonstration
        result = input_tensor * 2  # Multiply by 2
        for i in range(3):
            result = result + input_tensor  # Add input_tensor to result
        return result

# Input tensors with explicit values
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]),  # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]),  # 4D tensor
    torch.tensor([[7, 8, 9], [10, 11, 12]]),  # 2D tensor
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]])  # 3D tensor
]

if __name__ == "__main__":
    pt_module = PtModule()
    for tensor in input_tensors:
        output = pt_module.forward(tensor)
        print(f"Input:\n{tensor}\nOutput:\n{output}\n")