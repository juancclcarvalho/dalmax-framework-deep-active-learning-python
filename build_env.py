from torchvision import transforms
from dataset import DaninhasDatasetHandler
from data import get_DaninhasDataset
from models_dl import Net, DaninhasModel
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
                             LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                             KMeansSampling, KCenterGreedy, BALDDropout, \
                             AdversarialBIM, AdversarialDeepFool

params = {'DaninhasDataset':
              {'n_epoch': 10, 
               'train_args':{'batch_size': 512, 'num_workers': 4},
               'test_args':{'batch_size': 512, 'num_workers': 4},
               'optimizer_args':{'lr': 0.05, 'momentum': 0.3}}
          }

def get_handler(name):
    if name == 'DaninhasDataset':
        return DaninhasDatasetHandler
    else:
        raise NotImplementedError

def get_dataset(name):
    if name == 'DaninhasDataset':
        return get_DaninhasDataset(get_handler(name), "DATA/daninhas_full/")
    else:
        raise NotImplementedError
        
def get_net(name, device):
    if name == 'DaninhasDataset':
        return Net(DaninhasModel, params[name], device)
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