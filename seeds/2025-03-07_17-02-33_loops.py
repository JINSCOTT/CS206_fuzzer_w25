import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        
        for tensor in inputs:
            # Performing basic mathematical operations
            sum_tensor = tensor.sum()
            mean_tensor = tensor.mean()
            max_tensor = tensor.max()
            min_tensor = tensor.min()
            
            # Storing results in a list
            results.append({
                'sum': sum_tensor.item(),
                'mean': mean_tensor.item(),
                'max': max_tensor.item(),
                'min': min_tensor.item()
            })
        
        return results

# Input tensors
input_tensors = [
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),
    torch.tensor([[5, 6, 7], [8, 9, 10]]),
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]]),
    torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors)
    print(outputs)