import os
from pathlib import Path

'''
eu tenho um caminho de pasta: path/folder1
Eu quero retornar apenas o folder1. Porem, se eu uso o os.path e uso o base name funciona. Mas nao funciona no caso que 
tem uma barra no final. Entao, eu preciso retornar exatamente o nome da pasta, sem a barra no final.
'''
path = "path/folder1"
print(os.path.basename(path))
print(os.path.basename(path.rstrip("/")))