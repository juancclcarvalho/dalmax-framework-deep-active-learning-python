# Usage example: python3 mytest.py results/active_learning/uncertainty_sampling/selected_images/
import os
import sys

def count_files(dir_path):
    count_files_all_folder = 0
    for root, dirs, files in os.walk(dir_path):
        for dir in dirs:
            count_files_folder = len(os.listdir(os.path.join(root, dir)))
            print(f'{dir}: {count_files_folder}')
            count_files_all_folder += count_files_folder
    print(f'Total: {count_files_all_folder}')
    print(f'Average: {count_files_all_folder / len(os.listdir(dir_path))}')
if __name__ == '__main__':
    count_files(sys.argv[1])