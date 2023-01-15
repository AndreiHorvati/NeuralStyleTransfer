from utils import StyleTransferSettings
from image_loader import ImageLoader
from style_transfer import StyleTransfer
import matplotlib.pyplot as plt


class Menu:
    choices = ['1', '2', '3']

    @staticmethod
    def print_model_menu():
        print("Please select the model you want to use:")
        print("1.............VGG-16")
        print("2.............ResNet-50")
        print("3.............Inception-v3")
        print("0.............Exit")
        print(">>", end=" ")

    @staticmethod
    def run():
        print("Welcome to our Neural Style Transfer App!\n")
        while True:
            Menu.print_model_menu()
            model_choice = input().strip()

            if model_choice == '0':
                break
            elif model_choice in Menu.choices:
                print("\nPlease select the number of steps you want to iterate:")
                print(">>", end=" ")

                try:
                    number_of_steps = int(input().strip())
                    style_transfer_settings = StyleTransferSettings(model_choice, number_of_steps)
                    style_transfer = StyleTransfer(style_transfer_settings)

                    content_image_tensor = ImageLoader.load_image("./images/landscape.jpg")
                    style_image_tensor = ImageLoader.load_image("./images/vangogh.png")

                    output_image = style_transfer.style_transfer(content_image_tensor, style_image_tensor)
                    Menu.show_image(output_image)

                except ValueError as error:
                    print(error)
                    print("\nThis is not a valid number!\n")
            else:
                print("\nThis command does not exist!\n")

    @staticmethod
    def show_image(image_tensor):
        plt.figure()

        image = ImageLoader.unload_image(image_tensor)
        plt.imshow(image)
        plt.pause(0.001)

        plt.ioff()
        plt.show()
