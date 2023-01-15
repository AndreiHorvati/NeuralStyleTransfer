import ssl
import torch
import torch.nn as nn
from utils import StyleTransferUtils
from normalization import Normalization
from loss_functions import ContentLoss, StyleLoss, CustomStyleLoss


ssl._create_default_https_context = ssl._create_unverified_context


class StyleTransfer:
    def __init__(self, settings):
        self.settings = settings

        self.content_losses = []
        self.style_losses = []

    def get_model(self, content_image, style_image):
        model = nn.Sequential(
            Normalization(StyleTransferUtils.normalization_mean,
                          StyleTransferUtils.normalization_std).to(StyleTransferUtils.device))

        content_index = 0
        style_index = 0

        layers_indexes = {}
        for layer in StyleTransferUtils.models[self.settings.model_choice].children():
            layers_indexes[type(layer)] = 0
        layers_indexes[nn.AvgPool2d] = 0

        for layer in StyleTransferUtils.models[self.settings.model_choice].children():
            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                layer = nn.AvgPool2d(kernel_size=layer.kernel_size)

            layers_indexes[type(layer)] += 1
            name = '{}_{}'.format(layer.__class__.__name__, layers_indexes[type(layer)])

            model.add_module(name, layer)
            print(name)

            if name in StyleTransferUtils.content_layers[self.settings.model_choice]:
                content = model(content_image).detach()
                content_loss = ContentLoss(content)
                content_index += 1
                model.add_module('content_loss_{}'.format(content_index), content_loss)
                self.content_losses.append(content_loss)

            if name in StyleTransferUtils.style_layers[self.settings.model_choice]:
                style = model(style_image).detach()
                style_loss = StyleLoss(style)
                style_index += 1
                model.add_module('style_loss_{}'.format(style_index), style_loss)
                self.style_losses.append(style_loss)

        for index in range(len(model) - 1, -1, -1):
            if isinstance(model[index], ContentLoss) or isinstance(model[index], StyleLoss):
                return model[:(index + 1)]

        return model

    def style_transfer(self, content_image, style_image, content_weight=1, style_weight=1000000):
        model = self.get_model(content_image, style_image)
        input_image = content_image.clone()

        input_image.requires_grad_(True)
        model.requires_grad_(False)

        optimizer = StyleTransferUtils.get_input_image_gradient_descent(input_image)

        iteration = [0]
        while iteration[0] <= self.settings.number_of_steps:
            def closure():
                with torch.no_grad():
                    input_image.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_image)
                style_score = 0
                content_score = 0

                for style in self.style_losses:
                    style_score += style.loss
                for content in self.content_losses:
                    content_score += content.loss

                loss = style_score * style_weight + content_score * content_weight
                loss.backward(retain_graph=True)

                iteration[0] += 1
                if iteration[0] % 2 == 0:
                    print("Iteration {}:".format(iteration[0]))
                    print('Loss : {:4f}'.format(loss.item()))
                    print()

                return loss

            optimizer.step(closure)

        with torch.no_grad():
            input_image.clamp_(0, 1)

        return input_image


# iau mai multe stiluri
# 