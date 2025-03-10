import torch
import torch.nn as nn

class PtModule(nn.Module):
    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, inputs):
        results = []
        for input_tensor in inputs:
            # Adding 10 to the tensor
            added = input_tensor + 10
            
            # Multiplying by 2
            multiplied = added * 2
            
            # A loop that calculates the sum of all elements in the tensor
            sum_value = torch.sum(multiplied)
            
            results.append(sum_value)
        
        return results

# Define input tensors
input_tensors = [
    torch.tensor([[1, 2, 3], [4, 5, 6]]),        # 2D tensor
    torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D tensor
    torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),  # 2D tensor
    torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]]), # 3D tensor
    torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]) # 4D tensor
]

# Main section to check if the script is runnable
if __name__ == "__main__":
    model = PtModule()
    outputs = model(input_tensors)
    print(outputs)