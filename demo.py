import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from utils.orchestrator import get_dataset, get_network_deep_learning, get_strategy


# My looger
from utils.LOGGER import get_logger, get_path_logger

# Global logger
logger = get_logger()
path_logger = get_path_logger()

def main(args):
    # Train model with PyTorch
    logger.warning(f"DalMax - Training the model with PyTorch...")

    # Create results directory
    dir_results = args.dir_results + f"{args.strategy_name}/"
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)

    logger.warning(vars(args))

    # fix random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False

    # device
    use_cuda = torch.cuda.is_available()
    logger.warning(f"use_cuda: {use_cuda}")
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.warning(f"device: {device}")

    ## 
    all_acc = []
    all_rounds = []

    dataset = get_dataset(args.dataset_name)                   # load dataset
    net = get_network_deep_learning(args.dataset_name, device)                   # load network
    strategy = get_strategy(args.strategy_name)(dataset, net, logger)  # load strategy


    # start experiment
    start_time = time.time()
    dataset.initialize_labels(args.n_init_labeled)

    # round 0 accuracy
    logger.warning("Round 0")
    strategy.info()
    # strategy.train_full()
    strategy.train()
    
    preds = strategy.predict(dataset.get_test_data())
    acc = dataset.cal_test_acc(preds)
    
    logger.warning(f"Round 0 testing accuracy: {acc}")
    # exit()
    
    all_acc.append(acc)
    all_rounds.append(0)

    for rd in range(1, args.n_round+1):
        logger.warning("==========================================================================>")
        logger.warning(f"Round {rd}")
        # query
        query_idxs = strategy.query(args.n_query)

        # update labels
        strategy.update(query_idxs)
        
        # info after query
        strategy.info()

        # train
        strategy.train()

        # calculate accuracy
        preds = strategy.predict(dataset.get_test_data())
        acc = dataset.cal_test_acc(preds)
        logger.warning(f"Round {rd} testing accuracy: {acc}")

        all_acc.append(acc)
        all_rounds.append(rd)

        # Print acc and rounds
        logger.warning(f'Local Accuracies: {all_acc}')
        logger.warning(f'Local Rounds: {all_rounds}')
        logger.warning("==========================================================================>")


    logger.warning(f'Final Accuracies: {all_acc}')
    logger.warning(f'Final Rounds: {all_rounds}')

    end_time = time.time()
    final_time_in_seconds = end_time - start_time
    logger.warning(f"Total time: {final_time_in_seconds} seconds")
    logger.warning("==========================================================================>")
    
    # Plotar um grafico de acc x rounds
    plt.plot(all_rounds, all_acc)
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.title(f"{args.strategy_name} - {args.dataset_name}")

    plt.savefig(f"{args.strategy_name}_{args.dataset_name}.pdf")
    # Move generated files to results directory
    os.rename(f"{args.strategy_name}_{args.dataset_name}.pdf", f"{dir_results}/{args.strategy_name}_{args.dataset_name}.pdf")
    os.rename(path_logger, dir_results + "/log-dalmax.log")

    # Save the model
    strategy.save_model(dir_results)

    ## SAVE DATA in JSON
    # Dados de configuração
    dados_config_results = {
        'dataset_name': args.dataset_name,
        'strategy_name': args.strategy_name,

        'n_init_labeled': args.n_init_labeled,
        'n_query': args.n_query,
        'n_round': args.n_round,
        'seed': args.seed,
        
        'data': all_acc,
        'rounds': all_rounds,
    }

    # Salvar dados em um arquivo JSON
    json_path = os.path.join(dir_results, "results.json")
    with open(json_path, "w") as json_file:
        json.dump(dados_config_results, json_file, indent=4)

    print(f"Dados salvos em {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_results', type=str, default='results/new_dalmax_train_10_epochs_50_n_query/', help='Results directory')

    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--n_init_labeled', type=int, default=100, help="number of init labeled samples")
    parser.add_argument('--n_query', type=int, default=10, help="number of queries per round")
    parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
    parser.add_argument('--dataset_name', type=str, default="CIFAR10", choices=["CIFAR10", "DANINHAS"], help="dataset")
    parser.add_argument('--strategy_name', type=str, default="RandomSampling", 
                        choices=["RandomSampling", 
                                "LeastConfidence", 
                                "MarginSampling", 
                                "EntropySampling", 
                                "LeastConfidenceDropout", 
                                "MarginSamplingDropout", 
                                "EntropySamplingDropout", 
                                "KMeansSampling",
                                "KCenterGreedy", 
                                "BALDDropout", 
                                "AdversarialBIM", 
                                "AdversarialDeepFool"], help="query strategy")
    args = parser.parse_args()

    main(args)
