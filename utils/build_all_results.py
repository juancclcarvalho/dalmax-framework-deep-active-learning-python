# Example usage: python utils/plot_results_dir.py --input_dir results/new_dalmax_balanceado_train_10_epochs_10_n_query

import os
import json
import glob
import enum
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsType(enum.Enum):
    ALL_ACC = "all_acc"
    ALL_PRECISION = "all_precision"
    ALL_RECALL = "all_recall"
    ALL_F1_SCORE = "all_f1_score"

    DICT = {'Accuracy': ALL_ACC, 'Precision': ALL_PRECISION, 'Recall': ALL_RECALL, 'F1-score': ALL_F1_SCORE}

def list_folders_with_pattern(initial_path, pattern="*"):
    """Lists folders within a given path that match a specific pattern.

    Args:
        initial_path: The initial directory path.
        pattern: The pattern to match (default is '*', matching all).

    Returns:
        A list of folder paths that match the pattern.
    """

    matching_files = glob.glob(os.path.join(initial_path, pattern))
    # Filter only folders
    folders = [file for file in matching_files if os.path.isdir(file)]
    folders = sorted(folders)
    return folders


def main(args):
    input_dir = args.input_dir
    pattern = args.pattern

    # Listar todas as pastas de input_dir
    folders = list_folders_with_pattern(input_dir, pattern)

    print(f"Pasta de entrada: {input_dir}")
    print(f"Padrão: {pattern}")
    print(f"Diretórios com o padrão {pattern}: {folders}")
    print(f"Total de diretórios: {len(folders)}")
    print("\n")

    for path_folder in folders:
        print(f"\nPath: {path_folder}")
        
        nq_folders = list_folders_with_pattern(path_folder, "NQ_*")
        for path_nq_folder in nq_folders:
            print(f"\nPath NQ: {path_nq_folder}")
            
            method_folders = list_folders_with_pattern(path_nq_folder, "*")
            data = {}
            data_config = {}
            local_rounds = []

            plot_results(input_dir, path_nq_folder,  method_folders)
            exit()

            for path_method_folder in method_folders:
                print(f"Path Method: {path_method_folder}")
                
                path_results_json = os.path.join(path_method_folder, "results.json")
                print(f"JSON Path: {path_results_json}")

                if not os.path.exists(path_results_json):
                    raise ValueError(f"Arquivo {path_results_json} não encontrado.")
                
                with open(path_results_json, "r") as json_file:
                    data_config = json.load(json_file)
                
                print(f"data Config: {data_config}")
                exit()
        print("\n")
    exit()

def save_dict_to_json(data, json_path):
        # SAVE JSON
        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        print(f"JSON with data experiment saved in {json_path}")

def plot_results(input_dir, path_nq_folder, folders):
    # CREATE DIR RESULTS
    nq_basename = os.path.basename(path_nq_folder)
    dir_results = f"{input_dir}/results/{nq_basename}/"
    os.makedirs(dir_results, exist_ok=True)

    data_strategies = {}
    data_settings = {}
    for folder in folders:
        json_path = os.path.join(folder, "results.json")
        
        if not os.path.exists(json_path):
            raise ValueError(f"File {json_path} not found.")
        
        local_data_json = {}
        with open(json_path, "r") as json_file:
            local_data_json = json.load(json_file)

        data_strategies[local_data_json['strategy_name']] = {
            MetricsType.ALL_ACC.value: local_data_json[MetricsType.ALL_ACC.value],
            MetricsType.ALL_PRECISION.value: local_data_json[MetricsType.ALL_PRECISION.value],
            MetricsType.ALL_RECALL.value: local_data_json[MetricsType.ALL_RECALL.value],
            MetricsType.ALL_F1_SCORE.value: local_data_json[MetricsType.ALL_F1_SCORE.value]
        }

        # Get the last settings data
        data_settings = local_data_json

    # DICT SETTINGS DATA TO SAVE IN JSON
    new_data_config = {}
    new_data_config['dataset_name'] = data_settings['dataset_name']
    new_data_config['n_init_labeled'] = data_settings['n_init_labeled']
    new_data_config['n_query'] = data_settings['n_query']
    new_data_config['n_round'] = data_settings['n_round']
    new_data_config['rounds'] = data_settings['rounds']
    new_data_config['dir_results'] = dir_results
    new_data_config['data'] = data_strategies

    # SAVE JSON
    save_dict_to_json(new_data_config, os.path.join(dir_results, "data_experiment.json"))
    
    # SAVE PLOT
    save_plot(new_data_config, is_show=False)

def save_plot(new_data_config, is_show=False):
    data = new_data_config['data']
    local_rounds = new_data_config['rounds']
    dir_results = new_data_config['dir_results']
    
    metrics = MetricsType.DICT.value
    for key, value in metrics.items():
    
        # SETTINGS PLOT
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8, 6))

        markers = ['o', '*', 's', 'D', '^', 'P', 'X']
        colors = sns.color_palette("husl", len(data))

        # VAR SETTINGS
        value = value
        ylabel = key

        for i, (method, values) in enumerate(data.items()):
            marker = markers[i % len(markers)]
            if method == 'RandomSampling':
                plt.plot(local_rounds, values[value], label=method, color=colors[i], marker=marker, markersize=8, linestyle='-')
            else:
                plt.plot(local_rounds, values[value], label=method, color=colors[i])
        
        plt.title("Model comparison", fontsize=14)
        plt.xlabel("Rounds", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(title="Models")
        plt.tight_layout()

        path_plot = os.path.join(dir_results, f"metric_{ylabel.lower()}_methods_comparison.pdf")
        plt.savefig(path_plot)

        print(f"Plot to metric {key} saved in {path_plot}")
        if is_show:
            plt.show()

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="results/", help="Input directory path.")
    parser.add_argument("--pattern", type=str, default="*", help="Pattern to match (default is '*', matching all).")

    args = parser.parse_args()

    main(args)


