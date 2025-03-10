import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = {}
        results['add'] = torch.add(inputs[0], inputs[1])
        results['sub'] = torch.sub(inputs[0], inputs[1])
        results['mul'] = torch.mul(inputs[0], inputs[1])
        results['div'] = torch.div(inputs[0], torch.add(inputs[1], 1e-08))
        results['gt'] = inputs[0] > inputs[1]
        results['lt'] = inputs[0] < inputs[1]
        return results
input_tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]]), torch.tensor([[[1, 2, 3], [4, 5, 6]]]), torch.tensor([[[7, 8, 9], [10, 11, 12]]]), torch.tensor([[True, False], [False, True]])]
if __name__ == '__main__':
    model = PtModule()
    outputs = model(input_tensors)
    print(outputs)