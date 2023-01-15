import torch
import torch.nn as nn
import torch.nn.functional as functional


class ContentLoss(nn.Module):
    def __init__(self, content):
        super(ContentLoss, self).__init__()

        self.content = content.detach()

    def forward(self, input):
        self.loss = functional.huber_loss(input, self.content)

        return input


class CustomStyleLoss(nn.Module):
    @staticmethod
    def get_tensor_sum(tensor):
        return torch.sum(tensor)

    def __init__(self, style):
        super(CustomStyleLoss, self).__init__()

        self.style = style
        self.loss = 0

    def forward(self, input):
        a1, b1, c1, d1 = input.size()
        a2, b2, c2, d2 = self.style.size()

        vectorized_input = input.view(a1 * b1, c1 * d1)
        vectorized_style = self.style.view(a2 * b2, c2 * d2)
        batch_size = 100

        for index in range(a1 * b1):
            input_chunks = torch.split(vectorized_input[index], batch_size)
            style_chunks = torch.split(vectorized_style[index], batch_size)

            input_average_tensor = torch.FloatTensor(
                [CustomStyleLoss.get_tensor_sum(tensor) / batch_size for tensor in input_chunks])
            style_average_tensor = torch.FloatTensor(
                [CustomStyleLoss.get_tensor_sum(tensor) / batch_size for tensor in style_chunks])

            self.loss += functional.mse_loss(input_average_tensor, style_average_tensor)

        return input


class StyleLoss(nn.Module):
    @staticmethod
    def gram_matrix(tensor):
        a, b, c, d = tensor.size()

        features = tensor.view(a * b, c * d)
        G = torch.mm(features, features.t())

        return G.div(a * b * c * d)

    def __init__(self, style):
        super(StyleLoss, self).__init__()

        self.style = StyleLoss.gram_matrix(style).detach()

    def forward(self, input):
        self.loss = functional.mse_loss(StyleLoss.gram_matrix(input), self.style)

        return input
