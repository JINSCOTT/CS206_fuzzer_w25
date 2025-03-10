import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, x):
        added = torch.add(x, 2)
        multiplied = torch.mul(x, 3)
        compared = x > 1
        subtracted = torch.sub(x, 4)
        divided = torch.div(x, 2)
        return (added, multiplied, compared, subtracted, divided)
input_tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]), torch.tensor([[1, -1, 2]], dtype=torch.float32), torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int32)]

def main():
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(f'Input Tensor:\n{input_tensor}\nOutputs:\n{output}\n')
if __name__ == '__main__':
    main()