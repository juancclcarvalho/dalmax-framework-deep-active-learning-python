# Usage example: python3 mytest.py results/active_learning/uncertainty_sampling/selected_images/
import os
import sys

def count_files(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for dir in dirs:
            print(f'{dir}: {len(os.listdir(os.path.join(root, dir)))}')

if __name__ == '__main__':
    count_files(sys.argv[1])