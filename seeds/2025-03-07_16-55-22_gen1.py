import torch

class PtModule:
    def __init__(self):
        pass

    def compute(self, input_tensor):
        # Example mathematical operations
        output = input_tensor * 2  # Doubling the input tensor values
        for i in range(output.numel()):
            output.view(-1)[i] += i  # Adding index to each element
        return output

# Input tensors, 3 to 4 dimensions
input_tensors = [
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]),  # 4D tensor
    torch.tensor([[[10, 20], [30, 40]], [[50, 60], [70, 80]]]),  # 3D tensor
    torch.tensor([[[[10, 20], [30, 40]], [[50, 60], [70, 80]]]]),  # 4D tensor
    torch.tensor([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])  # 3D tensor
]

if __name__ == "__main__":
    module = PtModule()
    for tensor in input_tensors:
        result = module.compute(tensor)
        print(result)