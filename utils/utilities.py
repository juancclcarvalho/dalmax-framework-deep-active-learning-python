import os
import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn as sns

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import time

import torch.backends.cudnn as cudnn
import time
from PIL import Image
import argparse
import warnings

# GLOBAL SETTINGS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def count_files(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for dir in dirs:
            print(f'{dir}: {len(os.listdir(os.path.join(root, dir)))}')

def plot_confusion_matrix(iter, test_labels, predictions, label_map, dir_results, is_show=True):
    cm = confusion_matrix(test_labels.argmax(axis=1), predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.keys(), yticklabels=label_map.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{dir_results}/iteration_{iter}_confusion_matrix.pdf')
    if is_show:
        plt.show()

def plot_metrics(iter, history,dir_results, metrics=['loss', 'accuracy'], is_show=True):
    weighted_history = history
    # Plot loss and accuracy of training
    plt.figure(figsize=(6, 4))
    plt.title('Loss Plot')
    plt.plot(weighted_history.history['loss'], label='loss', color=colors[0])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{dir_results}/iteration_{iter}_training_loss_plot.pdf')
    if is_show:
        print('Loss: ', weighted_history.history['loss'])
        print('Accuracy: ', weighted_history.history['accuracy'])
        plt.show()
    # Reset plt 
    plt.clf()

    # Reseta o plt para plotar a acurácia
    plt.figure(figsize=(6, 4))
    plt.title('Accuracy Plot')
    plt.plot(weighted_history.history['accuracy'], label='accuracy', color=colors[0])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{dir_results}/iteration_{iter}_training_accuracy_plot.pdf')
    if is_show:
        plt.show()

def augment_image(img_array):
    import tensorflow as tf
    # Aplciar 4 técnicas de data augmentation
    img_array = tf.image.flip_left_right(img_array)
    img_array = tf.image.rot90(img_array)
    img_array = tf.image.random_brightness(img_array, 0.2)
    img_array = tf.image.random_contrast(img_array, 0.2, 0.5)


# Função para carregar imagens
def load_images(data_dir, img_size=(32, 32)):
    print(f'\n\n------------------')
    print(f'Loading images from {data_dir}')
    print(f'Image size: {img_size}')
    print(f'------------------\n\n')

    images = []
    labels = []
    paths = []
    label_map = {name: idx for idx, name in enumerate(os.listdir(data_dir))}
    for label_name, label_idx in label_map.items():
        class_dir = os.path.join(data_dir, label_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = image.load_img(img_path, target_size=img_size)
            img_array = image.img_to_array(img)

            # Aplicar data augmentation aqui
            # img_array = augment_image(img_array)
            
            images.append(img_array)
            labels.append(label_idx)
            paths.append(img_path)
    return np.array(images), np.array(labels), label_map, paths

def denormalize_image(image, mean, std):
    """
    Desnormaliza uma imagem normalizada.
    :param image: Tensor normalizado
    :param mean: Lista de médias usadas na normalização
    :param std: Lista de desvios padrão usados na normalização
    :return: Tensor desnormalizado
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return image * std + mean

def plot_n_examples(N, train_loader, num_classes):
    class_counts = {cls: 0 for cls in range(num_classes)}
    selected_images = []
    selected_labels = []

    for inputs, labels in train_loader:
        for input, label in zip(inputs, labels):
            if class_counts[label.item()] < 10:
                selected_images.append(input)
                selected_labels.append(label)
                class_counts[label.item()] += 1
            if all(count >= 10 for count in class_counts.values()):
                break
        if all(count >= 10 for count in class_counts.values()):
            break

    selected_images = torch.stack(selected_images)
    selected_labels = torch.stack(selected_labels)

    INDEX = selected_labels[0].item()
    logger.warning(f"Nome do arquivo Classe selecionada: {train_loader.dataset.samples[INDEX][0]}")


    # Imprimir o nome original de cada imagem selecionada
    logger.warning(f'Selected {N} images:')
    for i, label in enumerate(selected_labels):
        logger.warning(f"Image {i+1}: {train_loader.dataset.samples[label.item()][0]}")
        pass

    # Parâmetros de normalização usados em transforms.Normalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    plt.figure(figsize=(10, 10))
    for i in range(N):
        plt.subplot(5, 5, i+1)
        image = denormalize_image(selected_images[i], mean, std)  # Desnormaliza a imagem
        image = image.permute(1, 2, 0).clip(0, 1)  # Permuta dimensões e clipe valores para [0, 1]
        plt.imshow(image)
        plt.title(train_loader.dataset.classes[selected_labels[i].item()])
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"selected_{N}_images_original.pdf")

    plt.figure(figsize=(10, 10))
    for i in range(N):
        plt.subplot(5, 5, i+1)
        plt.imshow(selected_images[i].permute(1, 2, 0))
        plt.title(train_loader.dataset.classes[selected_labels[i].item()])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"selected_{N}_images_transformed.pdf")

    # Imprimir a porcentagem de cada classe selecionada selected_images
    logger.warning("Porcentagem de cada classe selecionada:")
    for cls, count in class_counts.items():
        logger.warning(f"Class {cls}: {count} ({count / 10 * 100:.0f}%)")
