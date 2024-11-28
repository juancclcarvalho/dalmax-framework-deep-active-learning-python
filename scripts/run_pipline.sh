#!/bin/bash
# Example: bash scripts/run_pipline.sh 0 1
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
    # AdversarialBIM
    # AdversarialDeepFool
)

# Loop pelas estratégias
for strategy in "${strategies[@]}"; do
    echo "Running $strategy"
    CUDA_VISIBLE_DEVICES=$1 python demo.py --dataset_name=DANINHAS --strategy_name $strategy --n_query 10 --seed $2 && notify-send "Process finished" $strategy
    CUDA_VISIBLE_DEVICES=$1 python demo.py --dataset_name=DANINHAS --strategy_name $strategy --n_query 50 --seed $2 && notify-send "Process finished" $strategy
    CUDA_VISIBLE_DEVICES=$1 python demo.py --dataset_name=DANINHAS --strategy_name $strategy --n_query 100 --seed $2 && notify-send "Process finished" $strategy

done
