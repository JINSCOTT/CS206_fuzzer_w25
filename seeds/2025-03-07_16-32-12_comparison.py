import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()
        
    def forward(self, inputs):
        results = {}
        
        # Math operations
        results['sum'] = inputs[0] + inputs[1]
        results['difference'] = inputs[2] - inputs[3]
        results['product'] = inputs[1] * inputs[4]
        results['quotient'] = inputs[4] / (inputs[1] + 1e-5)  # Adding small constant to avoid division by zero
        results['power'] = inputs[0] ** 2
        
        # Comparison operations
        results['greater_than'] = inputs[0] > inputs[1]
        results['less_than'] = inputs[2] < inputs[3]
        results['equal'] = inputs[1] == inputs[4]
        
        return results

# Input tensors
input_tensors = [
    torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    torch.tensor([[5, 6], [7, 8]], dtype=torch.float32),
    torch.tensor([[9, 10, 11], [12, 13, 14]], dtype=torch.float32),
    torch.tensor([[15, 16], [17, 18]], dtype=torch.float32),
    torch.tensor([[19], [20]], dtype=torch.float32)
]

if __name__ == "__main__":
    model = PtModule()
    output = model(input_tensors)
    print(output)