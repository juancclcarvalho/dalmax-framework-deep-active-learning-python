import json
import os
import numpy as np
import csv

def create_csv_tables(dados, dir_results):
    # List of NQ configurations to process
    nq_configs = [
        "NQ_10_NIL_100_NR_10_NE_10",
        "NQ_50_NIL_100_NR_10_NE_10",
        "NQ_100_NIL_100_NR_10_NE_10"
    ]
    
    # Loop through each NQ configuration
    for nq_config in nq_configs:
        # Dictionary to hold lists of metrics for each method
        method_metrics = {}
        
        # Loop through each seed
        for seed, seed_data in dados.items():
            if nq_config in seed_data:
                methods = seed_data[nq_config]
                for method, metrics in methods.items():
                    if method not in method_metrics:
                        method_metrics[method] = {
                            'acc': [],
                            'precision': [],
                            'recall': [],
                            'f1_score': []
                        }
                    # Collect metric values
                    method_metrics[method]['acc'].append(metrics['all_acc'])
                    method_metrics[method]['precision'].append(metrics['all_precision'])
                    method_metrics[method]['recall'].append(metrics['all_recall'])
                    method_metrics[method]['f1_score'].append(metrics['all_f1_score'])
        
        # Prepare data for CSV
        csv_data = []
        # Write header
        header = ['method', 'acc', 'precision', 'recall', 'f1_score']
        csv_data.append(header)
        
        # Write method data
        for method, metrics in method_metrics.items():
            # Calculate mean and std for each metric
            acc_mean = np.mean(metrics['acc'])
            acc_std = np.std(metrics['acc'])
            precision_mean = np.mean(metrics['precision'])
            precision_std = np.std(metrics['precision'])
            recall_mean = np.mean(metrics['recall'])
            recall_std = np.std(metrics['recall'])
            f1_mean = np.mean(metrics['f1_score'])
            f1_std = np.std(metrics['f1_score'])
            
            # Format the values as "mean ± std"
            acc = f"{acc_mean:.4f} ± {acc_std:.4f}"
            precision = f"{precision_mean:.4f} ± {precision_std:.4f}"
            recall = f"{recall_mean:.4f} ± {recall_std:.4f}"
            f1_score = f"{f1_mean:.4f} ± {f1_std:.4f}"
            
            # Append the row
            csv_data.append([method, acc, precision, recall, f1_score])
        
        # Create directory if it doesn't exist
        config_dir = os.path.join(dir_results, nq_config)
        os.makedirs(config_dir, exist_ok=True)
        
        # Define the CSV file path
        csv_file = os.path.join(config_dir, "tablea_com_media.csv")
        
        # Write to CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerows(csv_data)

# Load the data from dados.json
with open('results/dalmax/daninhas_balanceado/data_results.json', 'r') as file:
    dados = json.load(file)

# Specify the output directory
dir_results = 'results'

# Call the function
create_csv_tables(dados, dir_results)