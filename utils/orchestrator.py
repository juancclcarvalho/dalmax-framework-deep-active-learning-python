from torchvision import transforms
from utils.dataset import DANINHAS_Hander, CIFAR10_Handler
from utils.data import get_DANINHAS, get_CIFAR10

from core.deep_learning import DeepLearning
from core.daninhas_model import DaninhasModelResNet50, DaninhasModelVitB16
from core.cifar10_model import CIFAR10Model

from core.query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
                             LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                             KMeansSampling, KCenterGreedy, BALDDropout, \
                             AdversarialBIM, AdversarialDeepFool

params = {
        'DANINHAS':
            {'n_epoch': 10,
            'n_drop': 5,
            'n_classes': 5,
            'train_args':{'batch_size': 256, 'num_workers': 4},
            'test_args':{'batch_size': 256, 'num_workers': 4},
            'optimizer_args':{'lr': 0.05, 'momentum': 0.3}
            },
        
        'CIFAR10':
            {'n_epoch': 20, 
            'n_drop': 10,
            'n_classes': 10,
            'train_args':{'batch_size': 64, 'num_workers': 1},
            'test_args':{'batch_size': 1000, 'num_workers': 1},
            'optimizer_args':{'lr': 0.05, 'momentum': 0.3}
            }
            
        }

def get_handler(name):
    if name == 'DANINHAS':
        return DANINHAS_Hander
    
    elif name == 'CIFAR10':
        return CIFAR10_Handler
    else:
        raise NotImplementedError

def get_dataset(name):
    if name == 'DANINHAS':
        return get_DANINHAS(get_handler(name), "DATA/daninhas_full/")
    elif name == 'CIFAR10':
        return get_CIFAR10(get_handler(name), "DATA/DATA_CIFAR10/")
    else:
        raise NotImplementedError
        
def get_network_deep_learning(name, device):
    if name == 'DANINHAS':
        return DeepLearning(DaninhasModelResNet50, params[name], device)
    elif name == 'CIFAR10':
            return DeepLearning(CIFAR10Model, params[name], device)
    else:
        raise NotImplementedError
    
def get_params(name):
    return params[name]

def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "LeastConfidence":
        return LeastConfidence
    elif name == "MarginSampling":
        return MarginSampling
    elif name == "EntropySampling":
        return EntropySampling
    elif name == "LeastConfidenceDropout":
        return LeastConfidenceDropout
    elif name == "MarginSamplingDropout":
        return MarginSamplingDropout
    elif name == "EntropySamplingDropout":
        return EntropySamplingDropout
    elif name == "KMeansSampling":
        return KMeansSampling
    elif name == "KCenterGreedy":
        return KCenterGreedy
    elif name == "BALDDropout":
        return BALDDropout
    elif name == "AdversarialBIM":
        return AdversarialBIM
    elif name == "AdversarialDeepFool":
        return AdversarialDeepFool
    else:
        raise NotImplementedError