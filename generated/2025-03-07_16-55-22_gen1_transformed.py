import torch

class PtModule:

    def __init__(self):
        pass

    def compute(self, input_tensor):
        output = torch.mul(input_tensor, 2)
        for i in range(output.numel()):
            output.view(-1)[i] += i
        return output
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]), torch.tensor([[[10, 20], [30, 40]], [[50, 60], [70, 80]]]), torch.tensor([[[[10, 20], [30, 40]], [[50, 60], [70, 80]]]]), torch.tensor([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])]
if __name__ == '__main__':
    module = PtModule()
    for tensor in input_tensors:
        result = module.compute(tensor)
        print(result)