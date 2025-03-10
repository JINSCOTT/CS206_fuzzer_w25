import torch

class PtModule:

    def __init__(self):
        pass

    def compute(self, input_tensor):
        output = torch.zeros(input_tensor.size(0), input_tensor.size(1))
        for i in range(input_tensor.size(0)):
            output[i] = torch.sum(torch.pow(input_tensor[i], 2), dim=0)
        return output
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]), torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]], [[7.0], [8.0], [9.0]]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])]
if __name__ == '__main__':
    pt_module = PtModule()
    for input_tensor in input_tensors:
        result = pt_module.compute(input_tensor)
        print(result)