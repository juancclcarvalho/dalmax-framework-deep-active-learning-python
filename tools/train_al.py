# Example usage: python tools/train_al.py --dir_train DATA/DATA_CIFAR10/train/ --dir_test DATA/DATA_CIFAR10/test/ --dir_results results/ --type uncertainty_sampling --batch_size 10 --iterations 5 --test_size 0.9

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

# Local imports
from utils.utilities import load_images, plot_metrics, plot_confusion_matrix
from core.model_dl import create_model, create_parallel_model
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
    
    # Verifica se o tipo de active learning é válido
    if args.type not in ['uncertainty_sampling', 'query_by_committee', 'diversity_sampling', 'core_set_selection', 'adversarial_sampling', 'reinforcement_learning_sampling', 'expected_model_change', 'bayesian_sampling']:
        raise ValueError('Active Learning type must be: uncertainty_sampling, query_by_committee, diversity_sampling, core_set_selection, adversarial_sampling, reinforcement_learning_sampling, expected_model_change or bayesian_sampling')
    
    
def main(args):
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

    # Setup dir results
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)
    
    print(f"Initializating DalMax")
    print(f"Deep Learning Type Model: {args.type}")
    print(f"Parameters: batch_size: {batch_size}, iterations: {iterations}, test_size: {test_size}")

    # DATASET
    # Load dataset and preprocess
    images, labels, label_map, paths_images = load_images(dir_train)
    print(f"Classes label_map train: {label_map}")
    # Norm
    images = images / 255.0
    labels = to_categorical(labels, num_classes=len(label_map))

    # Split data
    train_images, pool_images, train_labels, pool_labels = train_test_split(images, labels, test_size=test_size, random_state=42)
    
    print(f"Percentage of train images:")
    for label_name, label_idx in label_map.items():
        print(f"{label_name}: {np.mean(train_labels.argmax(axis=1) == label_idx) * 100:.2f}%")
    
    # MODEL
    # Create model
    if mult_gpu:
        model = create_parallel_model(input_shape=train_images.shape[1:], num_classes=len(label_map))
    else:
        model = create_model(input_shape=train_images.shape[1:], num_classes=len(label_map))

    # START TRAINING
    start_time = time.time()
    final_weighted_history = None

    for i in range(iterations):
        try: 
            print(f"Iteration {i+1}/{iterations}")
            print(f"Actual Train Size: {len(train_images)} Actual Pool Size: {len(pool_images)}")
            
            selected_al_idx = None

            # Diversity Sampling
            if type_active_learning == 'diversity_sampling':
                selected_al_idx = DalMaxSampler.diversity_sampling(pool_images, batch_size)
            # Query-by-Committee
            elif type_active_learning == 'query_by_committee':
                committee_models = [create_model(input_shape=train_images.shape[1:], num_classes=len(label_map)) for _ in range(3)]
                for cm in committee_models:
                    cm.fit(train_images, train_labels, epochs=1, verbose=1)
                selected_al_idx = DalMaxSampler.query_by_committee(committee_models, pool_images, batch_size)
            # Uncertainty Sampling
            elif type_active_learning == 'uncertainty_sampling':
                selected_al_idx = DalMaxSampler.uncertainty_sampling(model, pool_images, batch_size)
            # Core-Set Selection (K-Center)
            elif type_active_learning == 'core_set_selection': 
                selected_al_idx = DalMaxSampler.core_set_selection(model, pool_images, batch_size)
            # Adversarial Active Learning
            elif type_active_learning == 'adversarial_sampling':
                selected_al_idx = DalMaxSampler.adversarial_sampling(model, pool_images, batch_size)
            # Reinforcement Learning for Active Learning
            elif type_active_learning == 'reinforcement_learning_sampling':
                # TODO: Implementar no futuro próximo
                # Assumindo um agente RL inicializado
                # selected_al_idx = DalMaxSampler.reinforcement_learning_sampling(agent, model, pool_images, batch_size)
                pass
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

            # Atualizar conjuntos de treino e pool
            train_images = np.concatenate([train_images, pool_images[selected_idx]])
            train_labels = np.concatenate([train_labels, pool_labels[selected_idx]])
            pool_images = np.delete(pool_images, selected_idx, axis=0)
            pool_labels = np.delete(pool_labels, selected_idx, axis=0)

            
            print('Selected_idx: ', selected_idx)
            print(f"New Train Size: {len(train_images)} New Pool Size: {len(pool_images)}")
            
            # Salve todas as imagens selecionadas em suas devidas pastas em dir_results/selected_images
            if not os.path.exists(f'{dir_results}/selected_images'):
                os.makedirs(f'{dir_results}/selected_images')
            for idx in selected_idx:
                img = pool_images[idx]
                label = pool_labels[idx].argmax()
                img_path = paths_images[idx]
                img_name = img_path.split('/')[-1]
                img_class = list(label_map.keys())[label]
                img_dir = f'{dir_results}/selected_images/{img_class}'
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
                plt.imsave(f'{img_dir}/{img_name}', img)

            # Treinar o modelo
            weighted_history = model.fit(train_images, train_labels, epochs=5, verbose=1)

            if final_weighted_history is None:
                final_weighted_history = weighted_history
            else:
                for key in final_weighted_history.history.keys():
                    final_weighted_history.history[key] += weighted_history.history[key]
        except Exception as e:
            print(f"Parando o processo pois os indices selecionados estão fora do range: {e}")
            break

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
    print(f"Active Learning Type: {args.type}")
    print("Done!")

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
    args = parser.parse_args()

    valid_args(args)
    main(args)