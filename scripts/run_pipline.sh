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

N_ROUND=10

# Loop pelas estratégias
for strategy in "${strategies[@]}"; do
    echo ""
    echo "Running $strategy"
    n_queries=(10 50 100)
    for n_query in "${n_queries[@]}"; do
        echo "Running $strategy with $n_query queries"
        CUDA_VISIBLE_DEVICES=$1 python demo.py --params_json params_dnb.json --dataset_name=DANINHAS --strategy_name $strategy --n_query $n_query --seed $2 --n_round $N_ROUND && notify-send "Process finished" $strategy
    done
done
