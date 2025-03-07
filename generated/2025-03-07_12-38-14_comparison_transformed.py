import torch
import torch.nn as nn

class PtModule(nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        add_result = torch.add(input_tensor[0], input_tensor[1])
        sub_result = torch.sub(input_tensor[1], input_tensor[2])
        mul_result = torch.mul(input_tensor[2], input_tensor[3])
        div_result = torch.div(input_tensor[3], torch.add(input_tensor[4], 1e-05))
        comparison_result = input_tensor[0] > input_tensor[4]
        return (add_result, sub_result, mul_result, div_result, comparison_result)
input_tensors = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[5, 6], [7, 8]], dtype=torch.float32), torch.tensor([[1, 1], [1, 1]], dtype=torch.float32), torch.tensor([[2, 3], [4, 5]], dtype=torch.float32), torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)]
if __name__ == '__main__':
    model = PtModule()
    results = model(input_tensors)
    for result in results:
        print(result)