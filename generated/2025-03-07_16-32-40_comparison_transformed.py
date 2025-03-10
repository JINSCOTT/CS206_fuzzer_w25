import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        add_result = torch.add(input_tensor, 5)
        sub_result = torch.sub(input_tensor, 3)
        mul_result = torch.mul(input_tensor, 2)
        div_result = torch.div(input_tensor, 4)
        greater_than = input_tensor > 1
        less_than = input_tensor < 10
        return (add_result, sub_result, mul_result, div_result, greater_than, less_than)
input_tensors = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32), torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32), torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    for (i, tensor) in enumerate(input_tensors):
        results = model(tensor)
        print(f'Results for input tensor {i}:')
        for result in results:
            print(result)