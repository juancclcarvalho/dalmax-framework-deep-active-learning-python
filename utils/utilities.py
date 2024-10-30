import os
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn as sns

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def count_files(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for dir in dirs:
            print(f'{dir}: {len(os.listdir(os.path.join(root, dir)))}')

def plot_confusion_matrix(test_labels, predictions, label_map, dir_results, is_show=True):
    cm = confusion_matrix(test_labels.argmax(axis=1), predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.keys(), yticklabels=label_map.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{dir_results}/confusion_matrix.pdf')
    if is_show:
        plt.show()

def plot_metrics(history,dir_results, metrics=['loss', 'accuracy'], is_show=True):
    weighted_history = history
    # Plot loss and accuracy of training
    plt.figure(figsize=(6, 4))
    plt.title('Loss Plot')
    plt.plot(weighted_history.history['loss'], label='loss', color=colors[0])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{dir_results}/training_loss_plot.pdf')
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
    plt.savefig(f'{dir_results}/training_accuracy_plot.pdf')
    if is_show:
        plt.show()

# Função para carregar imagens
def load_images(data_dir, img_size=(32, 32)):
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
            images.append(img_array)
            labels.append(label_idx)
            paths.append(img_path)
    return np.array(images), np.array(labels), label_map, paths