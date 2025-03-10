import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        add_result = torch.add(inputs[0], inputs[1])
        sub_result = torch.sub(inputs[1], inputs[2])
        mul_result = torch.mul(inputs[2], inputs[3])
        div_result = torch.div(inputs[4], torch.add(inputs[1], 1e-05))
        greater_than = inputs[0] > inputs[1]
        less_than = inputs[1] < inputs[2]
        return (add_result, sub_result, mul_result, div_result, greater_than, less_than)
input_tensors = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[5, 6], [7, 8]], dtype=torch.float32), torch.tensor([[9, 10], [11, 12]], dtype=torch.float32), torch.tensor([[13, 14], [15, 16]], dtype=torch.float32), torch.tensor([[17, 18], [19, 20]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors)
    for output in outputs:
        print(output)