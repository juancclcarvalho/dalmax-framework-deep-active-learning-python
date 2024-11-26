#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python demo.py --dataset_name=DANINHAS --strategy_name RandomSampling
CUDA_VISIBLE_DEVICES=1 python demo.py --dataset_name=DANINHAS --strategy_name LeastConfidence
CUDA_VISIBLE_DEVICES=0 python demo.py --dataset_name=DANINHAS --strategy_name MarginSampling
CUDA_VISIBLE_DEVICES=1 python demo.py --dataset_name=DANINHAS --strategy_name EntropySampling
CUDA_VISIBLE_DEVICES=0 python demo.py --dataset_name=DANINHAS --strategy_name LeastConfidenceDropout
CUDA_VISIBLE_DEVICES=1 python demo.py --dataset_name=DANINHAS --strategy_name MarginSamplingDropout
CUDA_VISIBLE_DEVICES=0 python demo.py --dataset_name=DANINHAS --strategy_name EntropySamplingDropout
CUDA_VISIBLE_DEVICES=1 python demo.py --dataset_name=DANINHAS --strategy_name KMeansSampling
CUDA_VISIBLE_DEVICES=0 python demo.py --dataset_name=DANINHAS --strategy_name KCenterGreedy
CUDA_VISIBLE_DEVICES=1 python demo.py --dataset_name=DANINHAS --strategy_name BALDDropout
CUDA_VISIBLE_DEVICES=0 python demo.py --dataset_name=DANINHAS --strategy_name AdversarialBIM
CUDA_VISIBLE_DEVICES=1 python demo.py --dataset_name=DANINHAS --strategy_name AdversarialDeepFool