#!/bin/bash

# Lista de estratégias
strategies=(
    RandomSampling
    LeastConfidence
    MarginSampling
    EntropySampling
    LeastConfidenceDropout
    MarginSamplingDropout
    EntropySamplingDropout
    KMeansSampling
    KCenterGreedy
    BALDDropout
    AdversarialBIM
    AdversarialDeepFool
)

# Loop pelas estratégias
for strategy in "${strategies[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python demo.py --dataset_name=DANINHAS --strategy_name="$strategy"
done
