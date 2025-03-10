import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for input_tensor in inputs:
            # Example math operations
            addition = input_tensor + 2
            multiplication = input_tensor * 3
            average = torch.mean(input_tensor)
            results.append((addition, multiplication, average))
        return results

input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.float32),
    torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32),
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[[[1]], [[2]]], [[[3]], [[4]]]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors)
    for i, output in enumerate(outputs):
        print(f"Output for input tensor {i}:\nAddition:\n{output[0]}\nMultiplication:\n{output[1]}\nAverage:\n{output[2]}\n")