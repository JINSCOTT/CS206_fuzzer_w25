import torch

class PtModule:
    def __init__(self):
        pass
    
    def forward(self, inputs):
        results = []
        for tensor in inputs:
            result = torch.sum(tensor) * 0.1  # Simple operation: sum and scaling
            results.append(result)
        return results

# Input tensors definition
input_tensors = [
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),          # 3D tensor
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # 4D tensor
    torch.tensor([[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]]),                # 3D tensor
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]]),       # 4D tensor
    torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]])             # 4D tensor
]

if __name__ == "__main__":
    pt_module = PtModule()
    outputs = pt_module.forward(input_tensors)
    for i, output in enumerate(outputs):
        print(f"Output for input tensor {i+1}: {output.item()}")