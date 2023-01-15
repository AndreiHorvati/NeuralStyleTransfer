import torch
import torchvision.transforms as transforms
from PIL import Image
from utils import StyleTransferUtils


class ImageLoader:
    image_size = 512
    image_transformations = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    unloader = transforms.ToPILImage()

    @staticmethod
    def load_image(image_path):
        image = Image.open(image_path)
        image = ImageLoader.image_transformations(image).unsqueeze(0)

        return image.to(StyleTransferUtils.device, torch.float)

    @staticmethod
    def unload_image(image_tensor):
        image = image_tensor.cpu().clone()
        image = image.squeeze(0)
        image = ImageLoader.unloader(image)

        return image
