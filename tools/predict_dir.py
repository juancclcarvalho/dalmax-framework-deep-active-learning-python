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
    logger.warning(f"DalMax - Predicting the images on folder...")

    path_model = args.path_model
    data_dir_test = args.data_dir_test
    img_size = args.img_size

    # List of classes
    classes = sorted(os.listdir(data_dir_test))

    logger.warning("----------------------------------------------------------")
    logger.warning("Args: ")
    logger.warning(f"path_model={path_model}")
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

    logger.warning("Evaluation on the test set: " + data_dir_test)
    correct = 0
    total = 0

    for class_name in classes:
        class_path = os.path.join(data_dir_test, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    # Predição para a imagem
                    predicted_class, _ = myDL.predict_image(img_path, transform, classes, img_size)
                    
                    # Verifica se a predição está correta
                    if predicted_class == class_name:
                        correct += 1
                    total += 1

    # Calculating the accuracy
    accuracy = correct / total * 100
    logger.warning(f"Final accuracy on the test set: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DalMax - Framework for Deep Active Learning with PyTorch 1.7.1')

    parser.add_argument("--path_model", type=str, default="results/new_dalmax_train_10_epochs/model_final.pth", help="Path to the pre-trained model.")
    parser.add_argument("--data_dir_test", type=str, default="DATA/daninhas_full/test", help="Path to the test dataset.")
    parser.add_argument("--img_size", type=int, default=128, help="Size of the image to be predicted.")
    args = parser.parse_args()

    # Example of usage: python tools/predict_dir.py --path_model results/new_dalmax_train_10_epochs/model_final.pth --data_dir_test DATA/daninhas_full/test --img_size 128

    main(args)

