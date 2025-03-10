import torch

class PtModule:

    def __init__(self):
        pass

    def perform_operations(self, input_tensor):
        output = torch.add(input_tensor, 5)
        output = torch.mul(output, 2)
        return output
input_tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32), torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32), torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=torch.float32)]
if __name__ == '__main__':
    module = PtModule()
    for tensor in input_tensors:
        result = module.perform_operations(tensor)
        print(f'Input Tensor:\n{tensor}\nOutput Tensor:\n{result}\n')