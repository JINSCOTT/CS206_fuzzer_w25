import torch

class PtModule:
    def __init__(self):
        pass

    def perform_operations(self, input_tensor):
        # Example math operations
        output = input_tensor + 5  # Add 5 to every element
        output = output * 2        # Multiply every element by 2
        return output

input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),                # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32), # 4D tensor
    torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32),            # Another 2D tensor
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=torch.float32)    # 3D tensor
]

if __name__ == "__main__":
    module = PtModule()
    for tensor in input_tensors:
        result = module.perform_operations(tensor)
        print(f"Input Tensor:\n{tensor}\nOutput Tensor:\n{result}\n")