import torch

class PtModule:

    def __init__(self):
        pass

    def perform_operations(self, input_tensor):
        result = torch.mul(input_tensor, 2)
        for i in range(result.shape[0]):
            result[i] += torch.sum(input_tensor[i])
        return result
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]]), torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0]]]), torch.tensor([[[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], [[[7.0, 8.0, 9.0]], [[10.0, 11.0, 12.0]]], [[[13.0, 14.0, 15.0]], [[16.0, 17.0, 18.0]]]])]
if __name__ == '__main__':
    module = PtModule()
    for input_tensor in input_tensors:
        output = module.perform_operations(input_tensor)
        print(f'Input Tensor:\n{input_tensor}\nOutput Tensor:\n{output}\n')