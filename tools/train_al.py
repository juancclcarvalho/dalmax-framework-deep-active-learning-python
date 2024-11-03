# Example usage: python tools/train_al.py --dir_train DATA/DATA_CIFAR10/train/ --dir_test DATA/DATA_CIFAR10/test/ --dir_results results/ --type uncertainty_sampling --batch_size 10 --iterations 5 --test_size 0.9 --epochs 100 --mult_gpu True --use_gpu 0

# System imports
import os
import sys
import time
import argparse

# Data manipulation
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow and Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical # type: ignore

# Add path to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Definir a semente para o gerador de números aleatórios do NumPy
# np.random.seed(43)

# Local imports
from utils.utilities import load_images, plot_metrics, plot_confusion_matrix
from core.model_dl import create_model, create_parallel_model, DQNAgent
from core.dalmax import DalMaxSampler

def valid_args(args):
     # Testes de validação
    if not os.path.exists(args.dir_train):
        raise ValueError('Train directory not found')
    
    if not os.path.exists(args.dir_test):
        raise ValueError('Test directory not found')
    
    if args.batch_size <= 0:
        raise ValueError('Batch size must be greater than 0')
    
    if args.iterations <= 0:
        raise ValueError('Iterations must be greater than 0')
    
    if args.test_size <= 0 or args.test_size >= 1:
        raise ValueError('Test size must be between 0 and 1')
    
    if args.epochs <= 0:
        raise ValueError('Epochs must be greater than 0')

    if args.use_gpu not in [0, 1]:
        raise ValueError('Use GPU must be 0 or 1')
    
    # Verifica se o tipo de active learning é válido
    if args.type not in ['random_sampling','uncertainty_sampling', 'query_by_committee', 'diversity_sampling', 'core_set_selection', 'adversarial_sampling', 'reinforcement_learning_sampling', 'expected_model_change', 'bayesian_sampling']:
        raise ValueError('Active Learning type must be: uncertainty_sampling, query_by_committee, diversity_sampling, core_set_selection, adversarial_sampling, reinforcement_learning_sampling, expected_model_change or bayesian_sampling')
    
def task_dalmax(args):

    # SETTINGS 
    # Vars from args
    dir_results = args.dir_results + f'/active_learning/{args.type}/'
    dir_train = args.dir_train

    batch_size = args.batch_size
    iterations = args.iterations
    test_size = args.test_size
    type_active_learning = args.type

    mult_gpu = args.mult_gpu
    use_gpu = args.use_gpu

    # Setup dir results
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)

    print("---------------------------------------------")
    print("Initializating DalMax")

    # DATASET
    # Load dataset and preprocess
    images, labels, label_map, paths_images = load_images(dir_train)
    images = images / 255.0
    labels = to_categorical(labels, num_classes=len(label_map))

    print(f"Classes label_map train: {label_map}")


    # Split data
    train_images, pool_images, train_labels, pool_labels = train_test_split(images, labels, test_size=test_size, random_state=42)

    # Remover todos os itens train_images e inserir em pool_images
    pool_images = np.concatenate([pool_images, train_images])
    pool_labels = np.concatenate([pool_labels, train_labels])
    train_images = np.empty((0,) + pool_images.shape[1:], dtype=pool_images.dtype)
    train_labels = np.empty((0,) + pool_labels.shape[1:], dtype=pool_labels.dtype)

    # Selecione as primeiras 10 imagens de cada classe para o conjunto do pool e insira no conjunto de treinamento. De modo que o conjuto de treinamento tenha extamente 10 imagens de cada classe
    for label_name, label_idx in label_map.items():
        idx_label = np.where(pool_labels.argmax(axis=1) == label_idx)[0][:10]
        train_images = np.concatenate([train_images, pool_images[idx_label]])
        train_labels = np.concatenate([train_labels, pool_labels[idx_label]])
        pool_images = np.delete(pool_images, idx_label, axis=0)
        pool_labels = np.delete(pool_labels, idx_label, axis=0)

    print("Percentage of train images:") # Do train_images. Deve imprimir 0% para todas as classes
    for label_name, label_idx in label_map.items():
        print(f"{label_name}: {len(np.where(train_labels.argmax(axis=1) == label_idx)[0])} ({np.mean(train_labels.argmax(axis=1) == label_idx) * 100:.2f}%)")
    # Methods that do not use the model
    methods_not_use_model = ['random_sampling', 'diversity_sampling','query_by_committee']
    model = None
    if type_active_learning not in methods_not_use_model:
        # MODEL
        # Create model
        if mult_gpu:
            model = create_parallel_model(input_shape=train_images.shape[1:], num_classes=len(label_map))
        else:
            model = create_model(input_shape=train_images.shape[1:], num_classes=len(label_map), mult_gpu=mult_gpu, use_gpu=use_gpu)

    # START TRAINING
    start_time = time.time()

    # Reinforcement Learning for Active Learning
    agent = None
    if type_active_learning == 'reinforcement_learning_sampling':
        # Parâmetros de entrada
        # Buscar a dimensão correta das imagens
        input_dim = train_images.shape[1:][0] * train_images.shape[1:][1] * train_images.shape[1:][2]
        print(f"Input Dim: {input_dim}")
        output_dim = 2   # número de ações possíveis (0 ou 1)

        # Inicializando o agente
        agent = DQNAgent(input_dim, output_dim)

    #for i in range(iterations):
    AUX = 0
    while AUX < iterations:
        i = AUX
        print(f"Starting iteration {i+1}/{iterations}")
        
        try: 
            print("\n\n---------------------------------------------")
            print(f"Iteration {i+1}/{iterations}")
            print(f"Actual Train Size: {len(train_images)}")
            print(f"Actual Pool Size: {len(pool_images)}")
                
            print("Percentage of train images:") # Do train_images. Deve imprimir 0% para todas as classes
            for label_name, label_idx in label_map.items():
                print(f"{label_name}: {len(np.where(train_labels.argmax(axis=1) == label_idx)[0])} ({np.mean(train_labels.argmax(axis=1) == label_idx) * 100:.2f}%)")
            
            # Se cada classe não tiver extamente a quantidade de imagens de  batch_size +10 imagens aumentar o número de iterações
            IS_BREAK = True
            for label_name, label_idx in label_map.items():
                if len(np.where(train_labels.argmax(axis=1) == label_idx)[0]) < batch_size + 10:
                    iterations += 1
                    print(f"Increasing iterations to {iterations} due to class {label_name} having less than {batch_size + 10} images")
                    # A classe X tem atualmente Y imagens, mas precisa de pelo menos Z imagens
                    print(f"The class {label_name} currently has {len(np.where(train_labels.argmax(axis=1) == label_idx)[0])} images, but needs at least {batch_size + 10} images")
                    IS_BREAK = False
                    break

            # Se cada classe de train forem iguais a batch_size + 10 PARAR
            if all([len(np.where(train_labels.argmax(axis=1) == label_idx)[0]) == batch_size + 10 for label_name, label_idx in label_map.items()]):
                print(f"Each class has exactly {batch_size + 10} images. Stopping iterations")
                break

            if IS_BREAK:
                break
            
            # If type_active_learning is in methods_not_use_model, do not train the model
            if type_active_learning not in methods_not_use_model:
                # Train model
                print("\nTraining main model")
                print(f"-----------------------------")
                model.fit(train_images, train_labels, epochs=10, verbose=1)
                print(f"-----------------------------")
            
            print(f"\nSelecting {batch_size} images from pool with {type_active_learning} method")
            selected_al_idx = None
            # Random Sampling
            if type_active_learning == 'random_sampling':
                # Select batch_size * iterations images from pool
                selected_al_idx = DalMaxSampler.random_sampling(pool_images, batch_size*iterations)
            # Diversity Sampling
            elif type_active_learning == 'diversity_sampling':
                selected_al_idx = DalMaxSampler.diversity_sampling(pool_images, batch_size)
            # Uncertainty Sampling
            elif type_active_learning == 'uncertainty_sampling':
                selected_al_idx = DalMaxSampler.uncertainty_sampling(model, pool_images, batch_size)
            # Query-by-Committee
            elif type_active_learning == 'query_by_committee':
                committee_models = [create_model(input_shape=train_images.shape[1:], num_classes=len(label_map)) for _ in range(3)]
                for cm in committee_models:
                    cm.fit(train_images, train_labels, epochs=1, verbose=1)
                selected_al_idx = DalMaxSampler.query_by_committee(committee_models, pool_images, batch_size)
            # Core-Set Selection (K-Center)
            elif type_active_learning == 'core_set_selection': 
                selected_al_idx = DalMaxSampler.core_set_selection(model, pool_images, batch_size)
            # Adversarial Active Learning
            elif type_active_learning == 'adversarial_sampling':
                selected_al_idx = DalMaxSampler.adversarial_sampling(model, pool_images, batch_size)
            # Reinforcement Learning for Active Learning
            elif type_active_learning == 'reinforcement_learning_sampling':
                # Assumindo um agente RL inicializado
                selected_al_idx = DalMaxSampler.reinforcement_learning_sampling(agent, model, pool_images, batch_size)
                print(f"Selected by RL: {selected_al_idx}")
            # Expected Model Change
            elif type_active_learning == 'expected_model_change':
                selected_al_idx = DalMaxSampler.expected_model_change(model, pool_images, batch_size)
            # Bayesian Sampling
            elif type_active_learning == 'bayesian_sampling':
                selected_al_idx = DalMaxSampler.bayesian_sampling(model, pool_images, batch_size)
            else:
                raise ValueError('Active Learning type must be uncertainty_sampling, query_by_committee or diversity_sampling')
                            
            # Escolher uma técnica por iteração (ou combinar)
            selected_idx = selected_al_idx  # Exemplo usando Uncertainty Sampling
            print(f"Selected new images: {len(selected_idx)} from pool")
            print(f"Selected index new images: {selected_idx}")

            print("\nUpdating Train and Pool sets")
            # Update train and pool sets
            train_images = np.concatenate([train_images, pool_images[selected_idx]])
            train_labels = np.concatenate([train_labels, pool_labels[selected_idx]])
            pool_images = np.delete(pool_images, selected_idx, axis=0)
            pool_labels = np.delete(pool_labels, selected_idx, axis=0)

            # Ensure the training set has exactly batch_size + 10 images per class
            for label_name, label_idx in label_map.items():
                idx_label = np.where(train_labels.argmax(axis=1) == label_idx)[0]
                print(f"Class {label_name} has {len(idx_label)} images")
                if len(idx_label) > batch_size + 10:
                    print(f"Removing {len(idx_label) - (batch_size + 10)} images from class {label_name}")
                    excess_idx = idx_label[batch_size + 10:]
                    pool_images = np.concatenate([pool_images, train_images[excess_idx]])
                    pool_labels = np.concatenate([pool_labels, train_labels[excess_idx]])
                    train_images = np.delete(train_images, excess_idx, axis=0)
                    train_labels = np.delete(train_labels, excess_idx, axis=0)
            
            print(f"New Train Size: {len(train_images)}")
            print(f"New Pool Size: {len(pool_images)}")
            print(f"Concluded iteration {i+1}/{iterations}. Next iteration...")
            print("---------------------------------------------")

            # If type_active_learning is random_sampling, break the loop
            if type_active_learning == 'random_sampling':
                break
            
            AUX += 1
        except Exception as e:
            print(f'Stopping iteration {i+1}/{iterations}: {e}')
            exit(1)
            break
    
    # Save all selected images in their respective folders in dir_results/selected_images
    if not os.path.exists(f'{dir_results}/selected_images'):
        print(f"Creating selected_images folder on {dir_results}/selected_images")
        os.makedirs(f'{dir_results}/selected_images')
    
    print(f"Saving selected images in {dir_results}/selected_images")
    for idx in range(len(train_images)):
        img = train_images[idx]
        label = train_labels[idx].argmax()
        img_class = list(label_map.keys())[label]
        img_dir = f'{dir_results}/selected_images/{img_class}'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img_name = f'{idx}.png'
        plt.imsave(f'{img_dir}/{img_name}', img)

    print(f"Images saved in {dir_results}/selected_images\n\n")
    
    end_time = time.time()
    text_time = f"Total time: {end_time - start_time:.2f} seconds"
    print(text_time)
    
    # Save time on infos file
    with open(f'{dir_results}/dalmax_time_process.txt', 'w') as f:
        f.write(f"{text_time}\n")
        f.write(f"Results saved in {dir_results}\n")
        f.write(f"Active Learning Task: {args.type}\n")
    
    print("Task DalMax Done!")
    print("---------------------------------------------")

    del model
    del train_images
    del pool_images
    del train_labels
    del pool_labels

def task_train(args):
    # SETTINGS 
    # Vars from args
    dir_results = args.dir_results + f'/active_learning/{args.type}/'
    dir_test = args.dir_test

    mult_gpu = args.mult_gpu
    use_gpu = args.use_gpu
    num_epochs = args.epochs

    # Setup dir results
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)
    if not os.path.exists(f'{dir_results}/selected_images'):
        raise ValueError('Selected not found')
    
    # Verifica se a pasta esta vazia
    if len(os.listdir(f'{dir_results}/selected_images/')) == 0:
        raise ValueError('Images not found in selected_images folder')
    
    print("---------------------------------------------")
    print("Initializating Task Train")

    # NEW TRAING TASK
    # DATASET
    # Load dataset and preprocess
    images, labels, label_map, paths_images = load_images(f'{dir_results}/selected_images/')
    images = images / 255.0
    labels = to_categorical(labels, num_classes=len(label_map))

    print(f"Classes label_map train: {label_map}")

    # Split data
    train_images = images
    train_labels = labels

    print("Percentage of train images:")
    for label_name, label_idx in label_map.items():
        print(f"{label_name}: {len(np.where(train_labels.argmax(axis=1) == label_idx)[0])} ({np.mean(train_labels.argmax(axis=1) == label_idx) * 100:.2f}%)")
    
    # MODEL
    # Create model
    if mult_gpu:
        model = create_parallel_model(input_shape=train_images.shape[1:], num_classes=len(label_map))
    else:
        model = create_model(input_shape=train_images.shape[1:], num_classes=len(label_map), mult_gpu=mult_gpu, use_gpu=use_gpu)
    
    # Treinar o modelo
    weighted_history = model.fit(train_images, train_labels, epochs=num_epochs, verbose=1)
    final_weighted_history = weighted_history
    start_time = time.time()
    # SAVE MODEL
    model.save(f'{dir_results}/final_{args.type}_al_model.h5')
    end_time = time.time()

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

    text_time = f"Total time: {end_time - start_time:.2f} seconds"
    print(text_time)
    
    # Save time on infos file
    with open(f'{dir_results}/dalmax_time_process.txt', 'w') as f:
        f.write(f"{text_time}\n")
        f.write(f"Results saved in {dir_results}\n")
        f.write(f"Active Learning Task: {args.type}\n")
    print(f"Results saved in {dir_results}")
    print(f"Active Learning Task: {args.type}")
    print("Task Train Done!")
    print("---------------------------------------------")

def main(args):
    print("Initializating Process")
    # Folders
    print(f"Train Directory: {args.dir_train}")
    print(f"Test Directory: {args.dir_test}")
    print(f"Results Directory: {args.dir_results}")
    # Active Learning
    print(f"Task Model: {args.type}")
    print("Parameters:")
    print(f"batch_size: {args.batch_size}")
    print(f"iterations: {args.iterations}")
    print(f"test_size: {args.test_size}")
    print(f"mult_gpu: {args.mult_gpu}")
    if not args.mult_gpu:
        print(f"use_gpu: {args.use_gpu}")
    print(f"epochs to train: {args.epochs}")

    if args.only_train:
        print("Task Only Train")
        task_train(args)
    else:
        print("Task DalMax + Train")
        task_dalmax(args)
        task_train(args)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DalMax - Framework for Deep Active Learning with TensorFlow 2.0')
    
    # Dataset directories
    parser.add_argument('--dir_train', type=str, default='DATA/DATA_CIFAR10/train/', help='Train dataset directory')
    parser.add_argument('--dir_test', type=str, default='DATA/DATA_CIFAR10/test/', help='Test dataset directory')
    parser.add_argument('--dir_results', type=str, default='results/', help='Results directory')

    # Type of Active Learning
    parser.add_argument('--type', type=str, default='uncertainty_sampling', help='Active Learning type')
    
    # Active Learning parameters
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size') # Quantidade de imagens selecionadas por vez do pool
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations')
    parser.add_argument('--test_size', type=float, default=0.9, help='Test size')
    parser.add_argument('--mult_gpu', type=bool, default=False, help='Use multiple GPUs')
    parser.add_argument('--use_gpu', type=int, default=0, help='Use GPU: 0 or 1')

    parser.add_argument('--epochs', type=int, default=10, help='Epochs size')
    # Only_train
    parser.add_argument('--only_train', type=bool, default=False, help='Only train the model')

    args = parser.parse_args()

    valid_args(args)
    main(args)