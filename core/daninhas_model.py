import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models

class DaninhasModel(nn.Module):
    def __init__(self):
        super(DaninhasModel, self).__init__()
        
        # Carregar a ResNet-50 pré-treinada
        resnet = models.resnet18(pretrained=True)
        
        # Remover a camada totalmente conectada final
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Todas as camadas até o penúltimo bloco
        
        # Número de features da penúltima camada
        num_features = resnet.fc.in_features  
        
        # Embedding layer para reduzir a dimensionalidade
        self.embedding_dim = 50
        self.embedding_layer = nn.Linear(num_features, self.embedding_dim)
        
        # Camada final para classificação em 5 classes
        self.classifier = nn.Linear(num_features, 5)

    def forward(self, x):
        # Extrair features da backbone ResNet
        features = self.feature_extractor(x)  # Saída: [batch_size, num_features, 1, 1]
        features = features.view(features.size(0), -1)  # Flatten para [batch_size, num_features]
        
        # Embedding: reduz para 50 dimensões
        e1 = self.embedding_layer(features)
        e1 = torch.relu(e1)
        
        # Saída final para classificação
        x = self.classifier(features)
        return x, e1

    def get_embedding_dim(self):
        return self.embedding_dim
