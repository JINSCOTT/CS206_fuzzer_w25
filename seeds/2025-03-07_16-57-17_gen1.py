import torch

class PtModule:
    def __init__(self):
        pass

    def forward(self, input_tensor):
        result = torch.zeros_like(input_tensor)
        for i in range(input_tensor.size(0)):
            result[i] = input_tensor[i] * 2 + 1  # A simple operation for demonstration
        return result

# Define input tensors
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32),
    torch.tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=torch.float32),
    torch.tensor([[[17, 18], [19, 20]], [[21, 22], [23, 24]]], dtype=torch.float32),
    torch.tensor([[[25, 26]], [[27, 28]], [[29, 30]]], dtype=torch.float32),
    torch.tensor([[[31]], [[32]], [[33]], [[34]]], dtype=torch.float32)
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    pt_module = PtModule()
    for input_tensor in input_tensors:
        output = pt_module.forward(input_tensor)
        print(f"Input Tensor:\n{input_tensor}\nOutput Tensor:\n{output}\n")