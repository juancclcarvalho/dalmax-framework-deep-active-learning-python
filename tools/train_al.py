# Example usage: python tools/train_al.py --dir_train DATA/DATA_CIFAR10/train/ --dir_test DATA/DATA_CIFAR10/test/ --dir_results results/ --type uncertainty_sampling --batch_size 10 --iterations 5 --test_size 0.9 --epochs 100 --seed 42 --mult_gpu True --use_gpu 0

# System imports
import os
import sys
import time
import argparse
import logging

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

# Configuração do logger
logger = logging.getLogger(__name__)  # Define o logger apenas para o seu módulo
logger.setLevel(logging.DEBUG)

# Criar um handler para escrever no arquivo de log
text_time_log = time.strftime('%Y-%m-%d-%H-%M-%S')
file_handler = logging.FileHandler(text_time_log + '-log-dalmax.log')
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
    if args.type not in ['random_sampling','uncertainty_sampling', 'query_by_committee', 'diversity_sampling', 'core_set_selection', 'adversarial_sampling', 'adversarial_sampling_ultra', 'reinforcement_learning_sampling', 'expected_model_change', 'bayesian_sampling']:
        raise ValueError('Active Learning type must be: uncertainty_sampling, query_by_committee, diversity_sampling, core_set_selection, adversarial_sampling, adversarial_sampling_ultra, reinforcement_learning_sampling, expected_model_change or bayesian_sampling')
    
    # Verifica se o seed é um inteiro válido > 0
    if not isinstance(args.seed, int) or args.seed <= 0:
        raise ValueError('Seed must be a positive integer')
    
def task_dalmax(args):
    # SETTINGS 
    # Vars from args
    dir_results = args.dir_results + f'/active_learning/{args.type}/'
    dir_train = args.dir_train
    dir_test = args.dir_test

    batch_size = args.batch_size
    iterations = args.iterations
    test_size = args.test_size
    type_active_learning = args.type

    mult_gpu = args.mult_gpu
    use_gpu = args.use_gpu
    seed = args.seed
    # Set seed
    np.random.seed(seed)

    # VARS
    EPOCHS_TRAIN_ACTIVE_LEARNING = 30
    INIT_IMAGES_PER_CLASS = 100
    all_accuracies = []
    all_train_sizes = []

    # Setup dir results
    if not os.path.exists(dir_results):
        logger.warning(f"Creating directory: {dir_results}")
        os.makedirs(dir_results)

    logger.warning("---------------------------------------------")
    logger.warning("Initializating DalMax")

    # DATASET
    # Load dataset and preprocess
    logger.warning("Loading dataset: Test")
    test_images, test_labels, label_map, paths_images = load_images(dir_test)
    test_images = test_images / 255.0
    test_labels = to_categorical(test_labels, num_classes=len(label_map))
    
    logger.warning(f"Loading dataset: Pool")
    images, labels, label_map, paths_images = load_images(dir_train)
    images = images / 255.0
    labels = to_categorical(labels, num_classes=len(label_map))

    logger.warning(f"Classes label_map train: {label_map}")
    train_images, pool_images, train_labels, pool_labels = train_test_split(images, labels, test_size=test_size, random_state=42)

    logger.warning("Moving all train images to pool")
    pool_images = np.concatenate([pool_images, train_images])
    pool_labels = np.concatenate([pool_labels, train_labels])
    train_images = np.empty((0,) + pool_images.shape[1:], dtype=pool_images.dtype)
    train_labels = np.empty((0,) + pool_labels.shape[1:], dtype=pool_labels.dtype)

    logger.warning(f"Selecting {INIT_IMAGES_PER_CLASS} images from pool to train")
    for label_name, label_idx in label_map.items():
        idx_label = np.where(pool_labels.argmax(axis=1) == label_idx)[0][:INIT_IMAGES_PER_CLASS]
        train_images = np.concatenate([train_images, pool_images[idx_label]])
        train_labels = np.concatenate([train_labels, pool_labels[idx_label]])
        pool_images = np.delete(pool_images, idx_label, axis=0)
        pool_labels = np.delete(pool_labels, idx_label, axis=0)

    logger.warning("Percentage of train images before active learning:")
    for label_name, label_idx in label_map.items():
        logger.warning(f"{label_name}: {len(np.where(train_labels.argmax(axis=1) == label_idx)[0])} ({np.mean(train_labels.argmax(axis=1) == label_idx) * 100:.2f}%)")

    # Reinforcement Learning for Active Learning
    agent = None
    if type_active_learning == 'reinforcement_learning_sampling':
        # Parâmetros de entrada
        logger.warning("Initializing Reinforcement Learning Agent")
        input_dim = train_images.shape[1:][0] * train_images.shape[1:][1] * train_images.shape[1:][2]
        logger.warning(f"Input Dim: {input_dim}")
        output_dim = 2   # número de ações possíveis (0 ou 1)

        # Inicializando o agente
        agent = DQNAgent(input_dim, output_dim)

    AUX = 0
    while AUX < iterations + 1:
        logger.warning("\n\n\n\n")

        start_time = time.time()
        i = AUX
        logger.warning(f"Starting iteration {i+1}/{iterations}")
        
        try: 
            model = None
            # Create model
            if mult_gpu:
                logger.warning("Creating parallel model")
                model = create_parallel_model(input_shape=train_images.shape[1:], num_classes=len(label_map))
            else:
                logger.warning("Creating model")
                model = create_model(input_shape=train_images.shape[1:], num_classes=len(label_map), mult_gpu=mult_gpu, use_gpu=use_gpu)

            logger.warning("\n---------------------------------------------")
            logger.warning(f"Iteration {i+1}/{iterations}")
            logger.warning(f"Actual Train Size: {len(train_images)}")
            logger.warning(f"Actual Pool Size: {len(pool_images)}")
                
            logger.warning("Percentage of train images:") # Do train_images. Deve imprimir 0% para todas as classes
            for label_name, label_idx in label_map.items():
                logger.warning(f"{label_name}: {len(np.where(train_labels.argmax(axis=1) == label_idx)[0])} ({np.mean(train_labels.argmax(axis=1) == label_idx) * 100:.2f}%)")
            
            # Train model
            logger.warning("\nTraining main model")
            logger.warning("-----------------------------")
            weighted_history = model.fit(train_images, train_labels, epochs=EPOCHS_TRAIN_ACTIVE_LEARNING, verbose=1)
            logger.warning("-----------------------------")
            
            # SAVE MODEL
            logger.warning("\nSaving model")
            model.save(f'{dir_results}/final_iteration_{AUX}_{args.type}_al_model.h5')

            # Plot training metrics
            logger.warning("\nPlotting training metrics")
            plot_metrics(AUX, weighted_history, dir_results, metrics=['loss', 'accuracy'], is_show=False)

            logger.warning("\nTesting model.")
            logger.warning("Predicting test images")
            predictions = model.predict(test_images).argmax(axis=1)
            logger.warning("Calculating accuracy")
            accuracy = accuracy_score(test_labels.argmax(axis=1), predictions)
            # Append accuracy and train size
            all_accuracies.append(accuracy)
            all_train_sizes.append(train_images.shape[0])
            
            # Save on file the final accuracy
            logger.warning(f"Final Test Accuracy Iteration {AUX}: {accuracy:.2f}")

            # Plot confusion matrix
            plot_confusion_matrix(iter=AUX, test_labels=test_labels, predictions=predictions, label_map=label_map, dir_results=dir_results, is_show=False)            
            logger.warning("---------------------------------------------")

            logger.warning(f"\nSelecting {batch_size} images from pool with {type_active_learning} method")
            
            selected_al_idx = None
            # Random Sampling
            if type_active_learning == 'random_sampling':
                logger.warning("Random Sampling")
                selected_al_idx = DalMaxSampler.random_sampling(pool_images, batch_size, seed)
            # Diversity Sampling
            elif type_active_learning == 'diversity_sampling':
                logger.warning("Diversity Sampling")
                selected_al_idx = DalMaxSampler.diversity_sampling(pool_images, batch_size, seed)
            # Uncertainty Sampling
            elif type_active_learning == 'uncertainty_sampling':
                logger.warning("Uncertainty Sampling")
                selected_al_idx = DalMaxSampler.uncertainty_sampling(model, pool_images, batch_size)
            # Query-by-Committee
            elif type_active_learning == 'query_by_committee':
                logger.warning("Query by Committee")
                committee_models = [create_model(input_shape=train_images.shape[1:], num_classes=len(label_map)) for _ in range(3)]
                for cm in committee_models:
                    cm.fit(train_images, train_labels, epochs=1, verbose=1)
                selected_al_idx = DalMaxSampler.query_by_committee(committee_models, pool_images, batch_size)
            # Core-Set Selection (K-Center)
            elif type_active_learning == 'core_set_selection': 
                logger.warning("Core Set Selection")
                selected_al_idx = DalMaxSampler.core_set_selection(model, pool_images, batch_size)
            # Adversarial Active Learning
            elif type_active_learning == 'adversarial_sampling':
                logger.warning("Adversarial Sampling")
                selected_al_idx = DalMaxSampler.adversarial_sampling(model, pool_images, batch_size)
            
            # Adversarial Active Learning Ultra
            elif type_active_learning == 'adversarial_sampling_ultra':
                logger.warning("Adversarial Sampling Ultra")
                selected_al_idx = DalMaxSampler.adversarial_sampling_ultra(model, pool_images, batch_size)

            # Reinforcement Learning for Active Learning
            elif type_active_learning == 'reinforcement_learning_sampling':
                logger.warning("Reinforcement Learning Sampling")
                # Assumindo um agente RL inicializado
                selected_al_idx = DalMaxSampler.reinforcement_learning_sampling(agent, model, pool_images, batch_size)
                logger.warning(f"Selected by RL: {selected_al_idx}")
            # Expected Model Change
            elif type_active_learning == 'expected_model_change':
                logger.warning("Expected Model Change")
                selected_al_idx = DalMaxSampler.expected_model_change_entropy(model, pool_images, batch_size)
            # Bayesian Sampling
            elif type_active_learning == 'bayesian_sampling':
                logger.warning("Bayesian Sampling")
                selected_al_idx = DalMaxSampler.bayesian_sampling(model, pool_images, batch_size)
            else:
                raise ValueError('Active Learning type must be uncertainty_sampling, query_by_committee or diversity_sampling')
                            
            # Escolher uma técnica por iteração (ou combinar)
            selected_idx = selected_al_idx  # Exemplo usando Uncertainty Sampling
            logger.warning(f"Selected new images: {len(selected_idx)} from pool")
            logger.warning(f"Selected index new images: {selected_idx}")

            logger.warning("\nUpdating Train and Pool sets")
            # Update train and pool sets
            train_images = np.concatenate([train_images, pool_images[selected_idx]])
            train_labels = np.concatenate([train_labels, pool_labels[selected_idx]])
            pool_images = np.delete(pool_images, selected_idx, axis=0)
            pool_labels = np.delete(pool_labels, selected_idx, axis=0)
            
            logger.warning(f"New Train Size: {len(train_images)}")
            logger.warning(f"New Pool Size: {len(pool_images)}")
            logger.warning(f"Concluded iteration {i+1}/{iterations}. Next iteration...")
            logger.warning("---------------------------------------------")
            
            # Logger all accuracies and train sizes
            logger.warning("All accuracies:")
            logger.warning(all_accuracies)
            logger.warning("All train sizes:")
            logger.warning(all_train_sizes) 

            AUX += 1

            end_time = time.time()
            logger.warning(f"Total time iteration: {end_time - start_time:.2f} seconds")
        
        except Exception as e:
            logger.warning(f'Stopping iteration {i+1}/{iterations}: {e}')
            break
    
    # Criar um gráfico com a acurácia e o tamanho do conjunto de treinamento
    plt.figure(figsize=(8, 6))
    plt.plot(all_train_sizes, all_accuracies, 'o-')
    plt.xlabel('Train Size')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy x Train Size: {args.type}')
    plt.tight_layout()
    plt.savefig(f'{dir_results}/iteration_final_accuracy_train_size.pdf')
    
    logger.warning("Task DalMax Done!")
    logger.warning("---------------------------------------------")

def main(args):
    start_time = time.time()
    logger.warning("Initializating Process")
    # Folders
    logger.warning(f"Train Directory: {args.dir_train}")
    logger.warning(f"Test Directory: {args.dir_test}")
    logger.warning(f"Results Directory: {args.dir_results}")
    # Active Learning
    logger.warning(f"Task Model: {args.type}")
    logger.warning("Parameters:")
    logger.warning(f"batch_size: {args.batch_size}")
    logger.warning(f"iterations: {args.iterations}")
    logger.warning(f"test_size: {args.test_size}")
    logger.warning(f"mult_gpu: {args.mult_gpu}")
    if not args.mult_gpu:
        logger.warning(f"use_gpu: {args.use_gpu}")
    logger.warning(f"epochs to train: {args.epochs}")
    logger.warning(f"seed: {args.seed}")

    if args.only_train:
        logger.warning("Task Only Train")
    else:
        logger.warning("Task DalMax + Train")
        task_dalmax(args)

    end_time = time.time()
    logger.warning(f"ALL TIME TO ALL TASKS: {end_time - start_time:.2f} seconds")
    
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

    # Seed for random
    parser.add_argument('--seed', type=int, default=42, help='Seed for random')

    args = parser.parse_args()

    valid_args(args)
    main(args)