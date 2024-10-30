# DalMax - Framework to Deep Active Learning Approaches
Repository of resources used in my doctoral studies in the area of ​​Deep Active Learning

![Deep Active Learning Framework](assets/active-learning-framework.png)

## Methodology Overview
The Framework to Deep Active Learning Approaches for Maximal Information Gain (DalMax) is a framework that aims to select the most informative samples for training a deep learning model. The DalMax framework is based on heuristic strategies that select the samples that maximize the information gain of the model. The DalMax framework is composed of this heuristic strategies:
- **Uncertainty Sampling**: Calculate the entropy or confidence margin to select samples.
- **Diversity Sampling**: Use clustering (e.g., K-means) to select samples that represent the diversity of the dataset.
- **Query by Committee (QBC)**: Train multiple models and select the samples where there is the greatest disagreement between them.
- **Core-Set Selection**: Use optimization methods such as K-Center to select subsets that effectively cover the data space.
- **Adversarial Active Learning**: Generate adversarial samples to identify model weaknesses.
- **Reinforcement Learning for Active Learning**: Apply RL to learn sample selection strategies. This is not implemented in this repository.
- **Expected Model Change**: Choose samples that, when labeled, are expected to cause the greatest change in the model.
- **Bayesian Active Learning**: Use Bayesian methods to model uncertainty and select samples that maximize information gain.

## Installation
To install the necessary dependencies, run the following command:

```bash
    conda create --name dalmax_gpu python=3.9
    conda activate dalmax_gpu
    pip install tensorflow numpy scikit-learn matplotlib
```

## Usage

### Dataset
This project uses the `cifar10` dataset that is zipped inside the `datasets` folder. It must be unzipped inside the `DATA` folder.
```bash
    unzip datasets/cifar10.zip -d DATA
```

### Training Deep Active Learning Models
To train the models, run the following command:
```bash
    python train_al.py --dir_train DATA/DATA_CIFAR10/train/ --dir_test DATA/DATA_CIFAR10/test/ --dir_results results/ --type uncertainty_sampling --batch_size 10 --iterations 5 --test_size 0.9
```
#### Parameters
- `dir_train`: Path to the training dataset.
- `dir_test`: Path to the test dataset.
- `dir_results`: Path to save the results.
- `type`: Active learning strategy. Options: `uncertainty_sampling`, `query_by_committee`, `diversity_sampling`, `core_set_selection`, `adversarial_sampling`, `reinforcement_learning_sampling`, `expected_model_change`, `bayesian_sampling`.
- `batch_size`: Number of samples to be selected in each iteration.
- `iterations`: Number of iterations.
- `test_size`: Proportion of the test dataset.

#### Results
The results are saved in the `results` folder. The following files are generated:

- `selected_images/`: Folder with the selected images in each iteration per class.
- `confusion_matrix.pdf`: Confusion matrix of Deep Active Learning model. This is valid only for the active learning strategies that use the test dataset.
- `final_accuracy.txt`: Final accuracy of the Deep Active Learning model. This is valid only for the active learning strategies that use the test dataset.
- `infos.txt`: Information about the training process.
- `query_by_committee_al_model.h5`: Deep Active Learning model. This is valid only for the active learning strategies that use the test dataset.
- `training_accuracy_plot.pdf`: Training accuracy plot.
- `training_loss_plot.pdf`: Training loss plot.

### Training Deep Learning Models
To train the models, you can select on RANDOM or ACTIVE mode. In the RANDOM mode, the model is trained with random samples. In the ACTIVE mode, the model is trained with the samples selected by the active learning strategy. The following command shows how to train the model in the RANDOM mode:

#### Random Mode
```bash
    python train.py --dir_train DATA/DATA_CIFAR10/train/ --dir_test DATA/DATA_CIFAR10/test/ --dir_results results/ --type random --epochs 10
```

#### Active Mode
```bash
    python train.py --dir_train DATA/DATA_CIFAR10/train/ --dir_test DATA/DATA_CIFAR10/test/ --dir_results results/ --type train --epochs 10
```

Parameters:
- `dir_train`: Path to the training dataset.
- `dir_test`: Path to the test dataset.
- `dir_results`: Path to save the results.
- `type`: Training mode. Options: `random`, `train`.
- `epochs`: Number of epochs.


## License

DalMax is under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

- Author: Mário de Araújo Carvalho
- Email: mariodearaujocarvalho@gmail.com
- Project: [https://github.com/MarioCarvalhoBr/dalmax-deep-active-learning-python](https://github.com/MarioCarvalhoBr/dalmax-deep-active-learning-python)
