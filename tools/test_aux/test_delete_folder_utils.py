# Usage example: python3 mytest.py results/active_learning/uncertainty_sampling/selected_images/
import os
import sys
import shutil
import random

def delete_files(dir_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Lê todos os arquivos de um diretório e salva 150 em uma nova pasta com o mesmo novo da pasta original. Salvar tudo em Output/pasta_original/
    # Lista todas as pastas de dir_path
    pasta = os.listdir(dir_path)

    # Qual pasta tem menos arquivos
    min_files = 0
    pasta_min_files = ''
    for pasta in pasta:
        # Lista todos os arquivos da pasta
        arquivos = os.listdir(dir_path + pasta + "/rgb/")
        print(f'{pasta}: {len(arquivos)}')
        if min_files == 0 or len(arquivos) < min_files:
            min_files = len(arquivos)
            pasta_min_files = pasta

    pasta = os.listdir(dir_path)
    
    print(f'Pasta com menos arquivos: {pasta_min_files} ({min_files})')

    # Para cada pasta em pasta
    for pasta in pasta:
        # Lista todos os arquivos da pasta
        arquivos = os.listdir(dir_path + pasta + "/rgb/")

        # Para cada arquivo em arquivos
        # Filtrar somente as imagens .jpg
        arquivos = [arquivo for arquivo in arquivos if arquivo.endswith('.jpg')]

        # Seleciona 150 arquivos aleatórios
        arquivos = random.sample(arquivos, min_files)
        # MAke new folder 
        if not os.path.exists(output_dir + pasta):
            os.makedirs(output_dir + pasta)
        for arquivo in arquivos:
            # Copia o arquivo para a pasta de output
            shutil.copyfile(dir_path + pasta + '/rgb/' + arquivo, output_dir + pasta + '/' + arquivo)
    print(f'Files moved to {output_dir}')
if __name__ == '__main__':
    delete_files(sys.argv[1], sys.argv[2])