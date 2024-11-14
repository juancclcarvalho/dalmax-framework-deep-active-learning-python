# Usage example: python tools/analysis_tools/count_files_folder.py DATA/daninhas_full/test
import os
import sys

def count_files(dir_path):
    count_files_all_folder = 0
    for root, dirs, files in os.walk(dir_path):
        for dir in dirs:
            count_files_folder = len(os.listdir(os.path.join(root, dir)))
            print(f'{root}/{dir}: {count_files_folder}')
            count_files_all_folder += count_files_folder
    print(f'Total: {count_files_all_folder}')
    print(f'Average: {count_files_all_folder / len(os.listdir(dir_path))}')
if __name__ == '__main__':
    count_files(sys.argv[1])