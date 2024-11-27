#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python demo.py --dataset_name=DANINHAS --strategy_name RandomSampling --n_query 50
CUDA_VISIBLE_DEVICES=1 python demo.py --dataset_name=DANINHAS --strategy_name LeastConfidence --n_query 50
CUDA_VISIBLE_DEVICES=1 python demo.py --dataset_name=DANINHAS --strategy_name MarginSampling --n_query 50
CUDA_VISIBLE_DEVICES=1 python demo.py --dataset_name=DANINHAS --strategy_name EntropySampling --n_query 50
CUDA_VISIBLE_DEVICES=1 python demo.py --dataset_name=DANINHAS --strategy_name LeastConfidenceDropout --n_query 50
CUDA_VISIBLE_DEVICES=1 python demo.py --dataset_name=DANINHAS --strategy_name MarginSamplingDropout --n_query 50
CUDA_VISIBLE_DEVICES=1 python demo.py --dataset_name=DANINHAS --strategy_name EntropySamplingDropout --n_query 50
CUDA_VISIBLE_DEVICES=1 python demo.py --dataset_name=DANINHAS --strategy_name KMeansSampling --n_query 50
CUDA_VISIBLE_DEVICES=1 python demo.py --dataset_name=DANINHAS --strategy_name KCenterGreedy --n_query 50
CUDA_VISIBLE_DEVICES=1 python demo.py --dataset_name=DANINHAS --strategy_name BALDDropout --n_query 50
CUDA_VISIBLE_DEVICES=1 python demo.py --dataset_name=DANINHAS --strategy_name AdversarialBIM
CUDA_VISIBLE_DEVICES=1 python demo.py --dataset_name=DANINHAS --strategy_name AdversarialDeepFool