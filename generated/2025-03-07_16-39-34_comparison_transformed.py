import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor1, input_tensor2):
        addition = torch.add(input_tensor1, input_tensor2)
        subtraction = torch.sub(input_tensor1, input_tensor2)
        multiplication = torch.mul(input_tensor1, input_tensor2)
        division = torch.div(input_tensor1, torch.add(input_tensor2, 1e-05))
        greater_than = input_tensor1 > input_tensor2
        less_than = input_tensor1 < input_tensor2
        equal_to = input_tensor1 == input_tensor2
        return (addition, subtraction, multiplication, division, greater_than, less_than, equal_to)
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[2.0, 3.0], [4.0, 5.0]]]), torch.tensor([[[6.0, 7.0], [8.0, 9.0]]]), torch.tensor([[[10.0]]]), torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])]
if __name__ == '__main__':
    module = PtModule()
    results = module(input_tensors[0], input_tensors[1])
    for result in results:
        print(result)