import torch

class PtModule(torch.nn.Module):

    def __init__(self):
        super(PtModule, self).__init__()

    def forward(self, input_tensor):
        output_tensor = input_tensor.clone()
        for i in range(output_tensor.size(0)):
            output_tensor[i] = torch.add(torch.mul(output_tensor[i], 2), 1)
        return output_tensor
input_tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), torch.tensor([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]), torch.tensor([[[17.0, 18.0, 19.0], [20.0, 21.0, 22.0]]]), torch.tensor([[[23.0, 24.0], [25.0, 26.0]], [[27.0, 28.0], [29.0, 30.0]], [[31.0, 32.0], [33.0, 34.0]]]), torch.tensor([[[35.0, 36.0], [37.0, 38.0]], [[39.0, 40.0], [41.0, 42.0]], [[43.0, 44.0], [45.0, 46.0]], [[47.0, 48.0], [49.0, 50.0]]])]
if __name__ == '__main__':
    model = PtModule()
    for input_tensor in input_tensors:
        output = model(input_tensor)
        print(output)