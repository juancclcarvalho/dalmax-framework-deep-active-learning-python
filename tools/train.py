'''
Example usage 
    - random: python train.py --dir_train DATA/DATA_CIFAR10/train/ --dir_test DATA/DATA_CIFAR10/test/ --dir_results results/ --type random --epochs 10
    - train: python train.py --dir_train results/active_learning/uncertainty_sampling/selected_images/ --dir_test DATA/DATA_CIFAR10/test/ --dir_results results/ --type train --epochs 10
'''

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# Local imports
from utils.utilities import load_images, plot_metrics, plot_confusion_matrix
from model_dl import create_model
import time

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def main(args):
    # SETTINGS 
    # Dataset paths
    dir_results = args.dir_results + f'/active_learning/{args.type}/'
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)

    dir_train = args.dir_train
    dir_test = args.dir_test
    # Deep Learning Loop
    num_epochs = args.epochs
    print(f"Deep Learning Type: {args.type}")

    print(f"Number of epochs: {num_epochs}")

    # DATASET
    # Load dataset and preprocess
    images, labels, label_map, paths_images = load_images(dir_train)
    print(f"Classes label_map train: {label_map}")

    images = images / 255.0
    labels = to_categorical(labels, num_classes=len(label_map))

    # Split data
    train_images = images
    train_labels = labels

    if args.type == 'random':
        # TODO: No futuro fazer esse processo automaticamente com base na quantidade de imagens do dataset do DAL
        new_train_images = []
        new_train_labels = []
        for label_name, label_idx in label_map.items():
            idxs = np.where(train_labels.argmax(axis=1) == label_idx)[0]
            """
            BASE: 
            8: 42 (11.93%)
            6: 22 (6.25%)
            7: 34 (9.66%)
            5: 45 (12.78%)
            1: 35 (9.94%)
            2: 33 (9.38%)
            3: 41 (11.65%)
            0: 43 (12.22%)
            4: 32 (9.09%)
            9: 25 (7.10%)
            """
            selected_idxs = np.random.choice(idxs, 
                                                  42 if label_name == '8' 
                                                else 22 if label_name == '6' 
                                                else 34 if label_name == '7' 
                                                else 45 if label_name == '5'
                                                else 35 if label_name == '1'
                                                else 33 if label_name == '2'
                                                else 41 if label_name == '3'
                                                else 43 if label_name == '0'
                                                else 32 if label_name == '4'
                                                else 25, replace=False)
            
            new_train_images.append(train_images[selected_idxs])
            new_train_labels.append(train_labels[selected_idxs])

        # Converter para numpy array
        new_train_images = np.concatenate(new_train_images)
        new_train_labels = np.concatenate(new_train_labels)

        # Atualiza a variável de treino
        train_images = new_train_images
        train_labels = new_train_labels

    print("Contagem de arquivos no diretório de treino:")
    for label_name, label_idx in label_map.items():
        print(f"{label_name}: {len(np.where(train_labels.argmax(axis=1) == label_idx)[0])} ({np.mean(train_labels.argmax(axis=1) == label_idx) * 100:.2f}%)")
    
    # MODEL
    # Create model
    model = create_model(input_shape=train_images.shape[1:], num_classes=len(label_map))

    # TRAINING
    start_time = time.time()
    
    final_weighted_history = None
    # Treinar o modelo
    weighted_history = model.fit(train_images, train_labels, epochs=num_epochs, verbose=1)
    final_weighted_history = weighted_history

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    # Save time on infos file
    with open(f'{dir_results}/infos.txt', 'w') as f:
        f.write(f"Total time: {end_time - start_time:.2f} seconds\n")
    
    # SAVE MODEL
    model.save(f'{dir_results}/{args.type}_al_model.h5')

    # Plot training metrics
    plot_metrics(final_weighted_history, dir_results, metrics=['loss', 'accuracy'], is_show=False)

    # EVALUATION
    # Avaliação final
    test_images, test_labels, label_map, paths_images = load_images(dir_test)
    print(f"Classes label_map test: {label_map}")
    test_images = test_images / 255.0
    test_labels = to_categorical(test_labels, num_classes=len(label_map))
    predictions = model.predict(test_images).argmax(axis=1)
    accuracy = accuracy_score(test_labels.argmax(axis=1), predictions)
    text_final = (f"Final Test Accuracy: {accuracy * 100:.2f}%")
    
    # Save on file the final accuracy
    with open(f'{dir_results}/final_accuracy.txt', 'w') as f:
        f.write(text_final)
    print(text_final)

    # Plot confusion matrix
    plot_confusion_matrix(test_labels=test_labels, predictions=predictions, label_map=label_map, dir_results=dir_results, is_show=False)
    print(f"Results saved in {dir_results}")
    print(f"Deep Learning Type: {args.type}")
    print("Done!")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Active Learning Example')
    
    # Dataset directories
    parser.add_argument('--dir_train', type=str, default='DATA/DATA_CIFAR10/train/', help='Train dataset directory')
    parser.add_argument('--dir_test', type=str, default='DATA/DATA_CIFAR10/test/', help='Test dataset directory')
    parser.add_argument('--dir_results', type=str, default='results/', help='Results directory')

    # Type of Active Learning
    parser.add_argument('--type', type=str, default='uncertainty_sampling', help='Active Learning type')
    
    parser.add_argument('--epochs', type=int, default=10, help='Epochs size')
    args = parser.parse_args()

    # Verifica se o tipo de active learning é válido
    if args.type not in ['random', 'train']:
        raise ValueError('Active Learning type must be uncertainty_sampling, query_by_committee or diversity_sampling')
    
    main(args)