import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        add_result = torch.add(input_tensor, 10)
        subtract_result = torch.sub(input_tensor, 5)
        multiply_result = torch.mul(input_tensor, 2)
        divide_result = torch.div(input_tensor, 3.0)
        greater_than = input_tensor > 5
        less_than = input_tensor < 2
        return (add_result, subtract_result, multiply_result, divide_result, greater_than, less_than)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]), torch.tensor([[10.0, 11.0], [12.0, 13.0]]), torch.tensor([1.0, 2.0, 3.0])]
if __name__ == '__main__':
    model = PtModule()
    for tensor in input_tensors:
        results = model(tensor)
        print('Input Tensor:\n', tensor)
        print('Results:\n', results)