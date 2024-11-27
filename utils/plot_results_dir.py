import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

import argparse

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("--dir_input", type=str, default="results/")

args = parser.parse_args()

dir_input = args.dir_input

# Listar todas as pastas de dir_input
folders = os.listdir(dir_input)

print(folders)

"""
# Cada folder tem um arquivo JSON com os dados: results.json
Esse arquivo tem a seguinte estrutura:
{
    "dataset_name": "DANINHAS",
    "strategy_name": "RandomSampling",
    "n_init_labeled": 100,
    "n_query": 50,
    "n_round": 10,
    "seed": 1,
    "data": [
        0.5223171889838556,
        0.5180436847103513,
        0.6011396011396012,
        0.19610636277302945,
        0.6471984805318138,
        0.26495726495726496,
        0.7188983855650523,
        0.6424501424501424,
        0.7103513770180437,
        0.7117758784425451,
        0.6429249762583096
    ],
    "rounds": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
    ]
}

# Monte um dicionário com os dados:
dados = {
    'RandomSampling': {'data': []},
    'MarginSampling': {'data': []},
}
"""

dados = {}
dados_config = {}
local_rounds = []

for folder in folders:
    # Ler o arquivo JSON
    json_path = os.path.join(dir_input, folder, "results.json")
    
    # Verifica se o arquivo existe: Se não, imprime um aviso e pula para a próxima iteração
    if not os.path.exists(json_path):
        print(f"Arquivo {json_path} não encontrado.")
        continue
    
    with open(json_path, "r") as json_file:
        dados_config = json.load(json_file)

    # Adicionar ao dicionário de dados
    dados[dados_config['strategy_name']] = {'data': dados_config['data']}

    # Adicionar os rounds
    local_rounds = dados_config['rounds']


dados_config['data'] = dados

print(f"\nDados: {dados}")
print(f"\nRounds: {local_rounds}")
print(f"\nConfiguração: {dados_config}")

dir_results = f"results/my_reports/{dados_config['dataset_name']}_seed_{dados_config['seed']}_n_init_{dados_config['n_init_labeled']}_n_query_{dados_config['n_query']}_n_round_{dados_config['n_round']}/"
os.makedirs(dir_results, exist_ok=True)


# Salvar dados em um arquivo JSON
json_path = os.path.join(dir_results, "experimento_1_dados.json")
with open(json_path, "w") as json_file:
    json.dump(dados_config, json_file, indent=4)

print(f"Dados salvos em {json_path}")

# Estilo do gráfico
sns.set_theme(style="whitegrid")

# Configurar o tamanho do gráfico
plt.figure(figsize=(8, 6))
# Lista de marcadores para os modelos
marcadores = ['o', '*', 's', 'D', '^', 'P', 'X']  # Marcadores possíveis
# Paleta de cores
cores = sns.color_palette("husl", len(dados))
# Plotar os dados
for i, (modelo, valores) in enumerate(dados.items()):
    # Excluir os 3 ultimos valores
    # valores['data'] = valores['data'][:-2]

    marcador = marcadores[i % len(marcadores)]  # Reutilizar marcadores se necessário
    plt.plot(local_rounds, valores['data'], label=modelo, color=cores[i], marker=marcador, markersize=8, linestyle='-')
# Configurações do gráfico
plt.title("Model comparison", fontsize=14)
plt.xlabel("Rounds", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend(title="Models")
plt.tight_layout()

plt.savefig(os.path.join(dir_results, "model_comparison_epochs.pdf"))
plt.show()
