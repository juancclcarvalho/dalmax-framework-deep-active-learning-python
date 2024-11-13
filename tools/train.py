'''
Example usage 
    - random: python tools/train.py --dir_train DATA/DATA_CIFAR10/train/ --dir_test DATA/DATA_CIFAR10/test/ --dir_results results/ --type random --epochs 10 --use_gpu 1 --mult_gpu True
    - train: python tools/train.py --dir_train results/active_learning/uncertainty_sampling/selected_images/ --dir_test DATA/DATA_CIFAR10/test/ --dir_results results/ --type train --epochs 10 --use_gpu 1 --mult_gpu True
'''
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import logging

# TensorFlow and Sklearn
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.metrics import accuracy_score

# Add path to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local imports
from utils.utilities import load_images, plot_metrics, plot_confusion_matrix
from core.model_dl import create_model, create_parallel_model
import time

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# Configuração do logger
logger = logging.getLogger(__name__)  # Define o logger apenas para o seu módulo
logger.setLevel(logging.DEBUG)

# Criar um handler para escrever no arquivo de log
text_time_log = time.strftime('%Y-%m-%d-%H-%M-%S')
PATH_LOG_FINAL = text_time_log + '-log-dalmax.log'
file_handler = logging.FileHandler(PATH_LOG_FINAL)
file_handler.setLevel(logging.DEBUG)

# Criar um handler para imprimir no console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Definir o formato do log
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Adicionar os handlers ao logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def valid_args(args):
     # Testes de validação
    if not os.path.exists(args.dir_train):
        raise ValueError('Train directory not found')
    
    if not os.path.exists(args.dir_test):
        raise ValueError('Test directory not found')
    
    if args.epochs <= 0:
        raise ValueError('Epochs must be greater than 0')
    
    if args.type not in ['random', 'train']:
        raise ValueError('Mode type must be: random or train')
    
    if args.use_gpu not in [0, 1]:
        raise ValueError('Use GPU must be 0 or 1')
    
    # img_size
    if args.img_size <= 0:
        raise ValueError('Image size must be greater than 0')
    
def main(args):
    try: 
        # SETTINGS 
        # Dataset paths
        dir_results = args.dir_results + f'/train/active_learning/{args.type}/'
        if not os.path.exists(dir_results):
            os.makedirs(dir_results)

        dir_train = args.dir_train
        dir_test = args.dir_test
        num_epochs = args.epochs
        mult_gpu = args.mult_gpu
        use_gpu = args.use_gpu
        img_size = args.img_size

        logger.warning(f"Deep Learning Type: {args.type}")
        logger.warning(f"Number of epochs: {num_epochs}")
        logger.warning(f"Multiple GPUs: {mult_gpu}")
        logger.warning(f"Use GPU: {use_gpu}")
        logger.warning(f"Image size: {img_size}")
        logger.warning(f"Train dataset directory: {dir_train}")
        logger.warning(f"Test dataset directory: {dir_test}")
        logger.warning(f"Results directory: {dir_results}")

        # DATASET
        # Load dataset and preprocess
        images, labels, label_map, paths_images = load_images(data_dir=dir_train, img_size=(img_size, img_size))
        logger.warning(f"Classes label_map train: {label_map}")

        images = images / 255.0
        labels = to_categorical(labels, num_classes=len(label_map))

        # Split data
        train_images = images
        train_labels = labels

        logger.warning("Contagem de arquivos no diretório de treino:")
        for label_name, label_idx in label_map.items():
            logger.warning(f"{label_name}: {len(np.where(train_labels.argmax(axis=1) == label_idx)[0])} ({np.mean(train_labels.argmax(axis=1) == label_idx) * 100:.2f}%)")
        
        # MODEL
        # Create model
        # Create model
        if mult_gpu:
            model = create_parallel_model(input_shape=train_images.shape[1:], num_classes=len(label_map))
        else:
            model = create_model(input_shape=train_images.shape[1:], num_classes=len(label_map), mult_gpu=mult_gpu, use_gpu=use_gpu)

        # TRAINING
        start_time = time.time()
        
        final_weighted_history = None
        # Treinar o modelo
        weighted_history = model.fit(train_images, train_labels, epochs=num_epochs, verbose=1)
        final_weighted_history = weighted_history

        end_time = time.time()
        logger.warning(f"Total time: {end_time - start_time:.2f} seconds")
        
        # Save time on infos file
        with open(f'{dir_results}/infos.txt', 'w') as f:
            f.write(f"Total time: {end_time - start_time:.2f} seconds\n")
        
        # SAVE MODEL
        model.save(f'{dir_results}/{args.type}_al_model.h5')

        # Plot training metrics
        plot_metrics("0000", final_weighted_history, dir_results, metrics=['loss', 'accuracy'], is_show=False)

        # EVALUATION
        # Avaliação final
        test_images, test_labels, label_map, paths_images = load_images(data_dir=dir_test, img_size=(img_size, img_size))
        logger.warning(f"Classes label_map test: {label_map}")
        test_images = test_images / 255.0
        test_labels = to_categorical(test_labels, num_classes=len(label_map))
        predictions = model.predict(test_images).argmax(axis=1)
        accuracy = accuracy_score(test_labels.argmax(axis=1), predictions)
        text_final = (f"Final Test Accuracy: {accuracy * 100:.2f}%")
        
        # Save on file the final accuracy
        with open(f'{dir_results}/final_accuracy.txt', 'w') as f:
            f.write(text_final)
        logger.warning(text_final)

        # Plot confusion matrix
        plot_confusion_matrix("0000", test_labels=test_labels, predictions=predictions, label_map=label_map, dir_results=dir_results, is_show=False)
        logger.warning(f"Results saved in {dir_results}")
        logger.warning(f"Deep Learning Type: {args.type}")
        logger.warning("Done!")
    except Exception as e:
        logger.error(f"Error: {e}")
        # Move PATH_LOG_FINAL to dir_results
        os.rename(PATH_LOG_FINAL, f'{dir_results}/{PATH_LOG_FINAL}')
        exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DalMax - Framework for Deep Active Learning with TensorFlow 2.0')
    
    # Dataset directories
    parser.add_argument('--dir_train', type=str, default='DATA/DATA_CIFAR10/train/', help='Train dataset directory')
    parser.add_argument('--dir_test', type=str, default='DATA/DATA_CIFAR10/test/', help='Test dataset directory')
    parser.add_argument('--dir_results', type=str, default='results/', help='Results directory')

    parser.add_argument('--type', type=str, default='train', help='Mode type')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs size')
    parser.add_argument('--mult_gpu', type=bool, default=False, help='Use multiple GPUs')
    parser.add_argument('--use_gpu', type=int, default=0, help='Use GPU: 0 or 1')
    # img_size
    parser.add_argument('--img_size', type=int, default=32, help='Image size')
    args = parser.parse_args()

    valid_args(args)
    
    main(args)