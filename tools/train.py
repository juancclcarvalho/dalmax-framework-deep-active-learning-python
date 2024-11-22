import os
import sys
import argparse
import warnings

# GLOBAL SETTINGS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# My looger
from utils.LOGGER import get_logger, get_path_logger

# My Dataset
from utils.dataset import Dataset

# My Deep Learning
from core.model_dl import ModelType, DeepLearning

# Global logger
logger = get_logger()
path_logger = get_path_logger()

# Função principal
def main(args):
    # Train model with PyTorch
    logger.warning(f"DalMax - Training the model with PyTorch...")

    # Create results directory
    dir_results = args.dir_results
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)

    # Get parameters
    data_dir = args.data_dir
    batch_size = args.batch_size
    img_size = args.img_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate

    logger.warning("----------------------------------------------------------")
    logger.warning("Args: ")
    logger.warning(f"dir_results={dir_results}")
    logger.warning(f"data_dir={data_dir}")
    logger.warning(f"batch_size={batch_size}")
    logger.warning(f"img_size={img_size}")
    logger.warning(f"num_epochs={num_epochs}")
    logger.warning(f"learning_rate={learning_rate}")
    logger.warning("----------------------------------------------------------\n")
    
    # Create the dataset
    myDataset = Dataset(data_dir, batch_size, img_size)

    # Load the dataset (train and test loaders)
    train_loader = myDataset.get_train_loader()
    test_loader = myDataset.get_test_loader()
    num_classes = myDataset.get_num_classes()

    # Print information about the dataset
    logger.warning("----------------------------------------------------------")
    logger.warning("Dataset information:")
    logger.warning(f"Dataset: {data_dir}")
    logger.warning(f"Dataset classes: {myDataset.get_name_classes()}")
    logger.warning(f"Num classes: {myDataset.get_num_classes()}\n")
    logger.warning("First image in the dataset train_loader:")
    logger.warning(f"Filename: {train_loader.dataset.samples[2][0]}")
    logger.warning(f"Class: {train_loader.dataset.classes[train_loader.dataset.samples[2][1]]}")
    logger.warning("----------------------------------------------------------\n")
       
    # Create the model
    myDL = DeepLearning(model_type=ModelType.RESNET50, num_classes=num_classes, learning_rate=learning_rate, logger=logger)

    # Train the model
    train_loss, train_acc = myDL.train_model(train_loader, num_epochs)

    # Evaluate the model
    __, __ = myDL.evaluate_model(test_loader)

    # Salva o modelo treinado
    # Save the model
    myDL.save_model("model_final.pth", dir_results)

    # Plot loss e accuracy
    myDL.plot_loss_accuracy(train_loss, train_acc, num_epochs, dir_results)

    # Move path_logger to dir_results
    os.rename(path_logger, dir_results + "/log-dalmax.log")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DalMax - Framework for Deep Active Learning with PyTorch 1.7.1')
    
    # Results directory
    parser.add_argument('--dir_results', type=str, default='results/new_dalmax_train_10_epochs/', help='Results directory')
    
    # Dataset directories
    parser.add_argument('--data_dir', type=str, default='DATA/daninhas_full/', help='Dataset directory')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--img_size', type=int, default=128, help='Image size')
    parser.add_argument('--num_epochs', type=int, default=2, help='Epochs size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    # Seed
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    args = parser.parse_args()

    # Example usage: python tools/train.py --data_dir DATA/daninhas_full/ --batch_size 256 --img_size 128 --num_epochs 10 --learning_rate 0.001
    main(args)