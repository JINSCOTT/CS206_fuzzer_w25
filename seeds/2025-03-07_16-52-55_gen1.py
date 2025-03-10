import torch

class PtModule:
    def __init__(self):
        pass

    def forward(self, inputs):
        results = []
        for i in range(len(inputs)):
            result = inputs[i] * 2 + 3
            results.append(result)
        return results

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]]),  # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]),  # 4D tensor
    torch.tensor([[5, 10, 15], [20, 25, 30]]),  # 2D tensor
    torch.tensor([[[10, 20], [30, 40]], [[50, 60], [70, 80]], [[90, 100], [110, 120]]])  # 3D tensor
]

if __name__ == "__main__":
    pt_module = PtModule()
    output = pt_module.forward(input_tensors)
    print(output)