import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input1, input2):
        addition = torch.add(input1, input2)
        subtraction = torch.sub(input1, input2)
        multiplication = torch.mul(input1, input2)
        division = torch.div(input1, torch.add(input2, 1e-05))
        elementwise_equal = input1 == input2
        elementwise_greater = input1 > input2
        elementwise_less = input1 < input2
        return (addition, subtraction, multiplication, division, elementwise_equal, elementwise_greater, elementwise_less)
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[2.0, 3.0], [4.0, 5.0]], [[6.0, 7.0], [8.0, 9.0]]]), torch.tensor([[[10.0, 11.0], [12.0, 13.0]]]), torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]), torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])]
if __name__ == '__main__':
    pt_module = PtModule()
    results = pt_module(input_tensors[0], input_tensors[1])
    print('Results of operations:')
    for result in results:
        print(result)