import argparse
import sys

class Parser():
    def parse(self):
        parser = argparse.ArgumentParser(description="Neural Network that detects ads on Images")


        if not "--config" in sys.argv:
            sys.argv.append("--config")
            sys.argv.append("defaultConfig.cfg")

        idx = sys.argv.index("--config")
        cfgFile = sys.argv[idx + 1]
        with open(cfgFile) as f:
            args = f.read().split()
            for i in range(0, len(args), 2):
                if not args[i] in sys.argv:
                    sys.argv.append(args[i])
                    sys.argv.append(args[i + 1])

        parser.add_argument("--config", type=str, help="Optional config file, from which args get loaded (default=defaultConfig.cfg)")

        model_opt = parser.add_argument_group("Model Options")
        model_opt.add_argument("--num_classes", type=int, help="Amount of colors (1 = Grayscale, 3 = RGB) (Train/Test mode)")
        model_opt.add_argument("--num_conv_layers", type=int, help="Amount of convolutional layers in model (Train mode)")
        model_opt.add_argument("--num_fully_layers", type=int, help="Amount of fully connected layers in model (Train mode)")
        model_opt.add_argument("--load_model", type=bool, help="Load model from 'model_path' and continue training (Train mode)")
        model_opt.add_argument("--model_path", type=str, help="Path to the saved .h5 model (Train/Test mode)")

        train_opt = parser.add_argument_group("Training options")
        train_opt.add_argument("--lr", type=float, help="Learning rate (Train mode)")
        train_opt.add_argument("--batch_size", type=int, help="Size of one batch (Train mode)")
        train_opt.add_argument("--num_epochs", type=int, help="Number of epochs (Train mode)")
        train_opt.add_argument("--tensorboard_logdir", type=str, help="Folder to save tensorboard logs (Train mode)")
        train_opt.add_argument("--model_dir", type=str, help="Folder where to save the models")

        data_opt = parser.add_argument_group("Dataset options")
        data_opt.add_argument("--image_width", type=int, help="Width of image (Train/Test mode)")
        data_opt.add_argument("--image_height", type=int, help="Height of image (Train/Test mode)")
        data_opt.add_argument("--num_channels", type=int, help="Number of color channels (1=BW,3=RGB)(Train/Test mode)")
        data_opt.add_argument("--shuffle", type=bool, help="Shuffle data before training? (Train mode)")
        data_opt.add_argument("--testing_path", type=str, help="Path to testing images (See README.md for folder structure) (Test mode)")
        data_opt.add_argument("--training_path", type=str, help="Path to training images (See README.md for folder structure) (Train mode)")
        data_opt.add_argument("--images_per_step", type=str, help="Number of images to load into memory")

        return parser.parse_args()

class FileParser(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            cfg = f.read().split()
            parser.parse_args(cfg, namespace)
            print("yikes")
