import torch

class PtModule:

    def __init__(self):
        pass

    def perform_operations(self, input_tensor):
        result = 0
        for i in range(input_tensor.size(0)):
            result += input_tensor[i].sum()
        return result
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32), torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=torch.float32), torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]], [[[5.0], [6.0]], [[7.0], [8.0]]]], dtype=torch.float32), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.int32), torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)]
if __name__ == '__main__':
    module = PtModule()
    for tensor in input_tensors:
        output = module.perform_operations(tensor)
        print(f'Output for tensor {tensor}: {output}')