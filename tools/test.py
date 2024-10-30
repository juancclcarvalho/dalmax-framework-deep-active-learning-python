# Usage: python test.py --dir_test DATA/DATA_CIFAR10/test/ --dir_results results/ --model results/active_learning_model.h5

# System imports
import os
import sys
import argparse
import time

# TensorFlow and Sklearn
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

# Add path to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local import
from utils.utilities import load_images

def valid_args(args):
    modal_path, dir_test, dir_results = args.model, args.dir_test, args.dir_results
    if not os.path.exists(modal_path):
        raise FileNotFoundError(f"Model not found: {modal_path}")
    if not os.path.exists(dir_test):
        raise FileNotFoundError(f"Test dataset not found: {dir_test}")
    if not os.path.exists(dir_results):
        raise FileNotFoundError(f"Results directory not found: {dir_results}")
    
def main(args):
    # Setup paths
    modal_path = args.model
    dir_test = args.dir_test
    dir_results = args.dir_results
    
    # Create if not exists
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)

    # Load the model
    model = load_model(modal_path)

    # Record the time
    init_time = time.time()
    
    # Avaliação final
    test_images, test_labels, label_map, __ = load_images(dir_test)
    test_images = test_images / 255.0
    test_labels = to_categorical(test_labels, num_classes=len(label_map))
    predictions = model.predict(test_images).argmax(axis=1)
    
    end_time = time.time()
    text_time = f"Total time: {end_time - init_time:.2f} seconds"
    
    accuracy = accuracy_score(test_labels.argmax(axis=1), predictions)
    text_final = f"Final Test Accuracy: {accuracy * 100:.2f}%"
   

    # Save on file the final accuracy
    with open(f'{dir_results}/final_accuracy.txt', 'w') as f:
        f.write(text_final)
    

    # Save time on infos file
    with open(f'{dir_results}/infos.txt', 'w') as f:
        f.write(text_time)
    
    print(text_time)
    print(text_final)
    print(f"Results saved on {dir_results}")
    print("Done!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Active Learning Model')
    parser.add_argument('--dir_test', type=str, default='DATA/DATA_CIFAR10/test/', help='Test dataset directory')
    parser.add_argument('--dir_results', type=str, default='results/', help='Results directory')
    parser.add_argument('--model', type=str, default='results/active_learning_model.h5', help='Model path')

    args = parser.parse_args()

    valid_args(args)
    main(args)