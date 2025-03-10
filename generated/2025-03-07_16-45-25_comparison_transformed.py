import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        add_result = torch.add(inputs[0], inputs[1])
        sub_result = torch.sub(inputs[2], inputs[3])
        mul_result = torch.mul(inputs[0], inputs[4])
        div_result = torch.div(inputs[3], torch.add(inputs[1], 1e-05))
        comp_result = inputs[2] > inputs[0]
        return (add_result, sub_result, mul_result, div_result, comp_result)
input_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[2.0, 3.0], [4.0, 5.0]]), torch.tensor([[0.5, 1.0], [1.5, 2.0]])]
if __name__ == '__main__':
    model = PtModule()
    results = model(input_tensors)
    for result in results:
        print(result)