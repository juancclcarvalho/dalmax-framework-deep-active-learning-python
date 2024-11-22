import os
import sys
import warnings
import argparse

from torchvision import transforms

# GLOBAL SETTINGS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# My looger
from utils.LOGGER import get_logger, get_path_logger
from core.model_dl import ModelType, DeepLearning

# Global logger
logger = get_logger()
path_logger = get_path_logger()

def main(args):
    # Predict the images
    logger.warning(f"DalMax - Predicting one image...")

    path_model = args.path_model
    path_image = args.path_image
    data_dir_test = args.data_dir_test
    img_size = args.img_size

    # List of classes
    classes = sorted(os.listdir(data_dir_test))

    logger.warning("----------------------------------------------------------")
    logger.warning("Args: ")
    logger.warning(f"path_model={path_model}")
    logger.warning(f"path_image={path_image}")
    logger.warning(f"data_dir_test={data_dir_test}")
    logger.warning(f"img_size={img_size}")
    logger.warning(f"classes={classes}")
    logger.warning("----------------------------------------------------------\n")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Tamanho compatível com ResNet-50
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create the model
    logger.warning("Re-create the model...")
    myDL = DeepLearning(model_type=ModelType.RESNET50, num_classes=len(classes), logger=logger)
    
    # Load the model
    logger.warning("Loading the model...")
    __ = myDL.load_model_pth(path_model)

    # Predict the image
    logger.warning(f"Predicting the image '{path_image}'...")
    predicted_class, confidence = myDL.predict_image(path_image, transform, classes, img_size)

    # Print the results
    logger.warning(f"Predicted class: {predicted_class}")
    logger.warning(f"Confidence: {confidence * 100:.4f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DalMax - Framework for Deep Active Learning with PyTorch 1.7.1')

    parser.add_argument("--path_model", type=str, default="results/new_dalmax_train_10_epochs/model_final.pth", help="Path to the pre-trained model.")
    parser.add_argument("--path_image", type=str, default="DATA/daninhas_full/test/DATASET_BRACHIARIA/ORTOFOTO_194_18573_25522_0.jpg", help="Path to the image to be predicted.")
    parser.add_argument("--data_dir_test", type=str, default="DATA/daninhas_full/test", help="Path to the test dataset.")
    parser.add_argument("--img_size", type=int, default=128, help="Size of the image to be predicted.")
    args = parser.parse_args()

    # Example of usage: python tools/predict_image.py --path_model results/new_dalmax_train_10_epochs/model_final.pth --path_image DATA/daninhas_full/test/DATASET_BRACHIARIA/ORTOFOTO_194_18573_25522_0.jpg --data_dir_test DATA/daninhas_full/test --img_size 128

    main(args)