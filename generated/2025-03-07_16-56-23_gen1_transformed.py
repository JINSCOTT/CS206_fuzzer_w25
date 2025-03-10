import torch

class PtModule:

    def __init__(self):
        pass

    def compute(self, input_tensor):
        output = input_tensor.clone()
        for i in range(output.size(0)):
            output[i] = torch.add(torch.mul(output[i], 2), 1)
        return output
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]]), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]]), torch.tensor([[[[1]], [[2]]], [[[3]], [[4]]], [[[5]], [[6]]]])]
if __name__ == '__main__':
    module = PtModule()
    for tensor in input_tensors:
        result = module.compute(tensor)
        print(f'Input Tensor:\n{tensor}\nOutput Tensor:\n{result}\n')