import torch

class PtModule:

    def __init__(self):
        pass

    def forward(self, input_tensor):
        output = input_tensor.clone()
        for i in range(output.shape[0]):
            output[i] = torch.add(output[i], torch.mul(2, i))
        return output
input_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32), torch.tensor([[9, 10, 11], [12, 13, 14]], dtype=torch.float32), torch.tensor([[[15], [16]], [[17], [18]], [[19], [20]]], dtype=torch.float32), torch.tensor([[[21, 22], [23, 24], [25, 26]], [[27, 28], [29, 30], [31, 32]]], dtype=torch.float32), torch.tensor([[33, 34], [35, 36], [37, 38]], dtype=torch.float32)]
if __name__ == '__main__':
    module = PtModule()
    for tensor in input_tensors:
        output_tensor = module.forward(tensor)
        print('Input Tensor:\n', tensor)
        print('Output Tensor:\n', output_tensor)
        print()