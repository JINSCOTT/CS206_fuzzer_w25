import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for tensor in inputs:
            # Perform some mathematical operations
            mean_value = torch.mean(tensor)
            sum_value = torch.sum(tensor)
            results.append((mean_value, sum_value))
        return results

# Define input tensors
input_tensors = [
    torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
    torch.tensor([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]),
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]),
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])
]

if __name__ == "__main__":
    module = PtModule()
    output = module(input_tensors)
    print(output)