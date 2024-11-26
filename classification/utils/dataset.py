import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image
import random

class Dataset():
    def __init__(self, data_dir="DATA/dataset/", batch_size=32, img_size=128, seed=42, n_init_labeled=100, is_active_learning=False, logger=None):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.n_init_labeled = n_init_labeled
        self.logger = logger
        self.seed = seed

        # PRE-CONFIGURE
        random.seed(seed)

        self.verify()
        self.logger.warning("==============================================================================")
        self.logger.warning("Building dataset with the following parameters:")
        self.logger.warning(f"Data directory: {data_dir}")
        self.logger.warning(f"Image size: {img_size}")
        self.logger.warning(f"Batch size: {batch_size}")
        self.logger.warning(f"Initial labeled pool size: {n_init_labeled}")
        self.logger.warning(f"Active learning: {is_active_learning}")

        self.labels = []
        self.classes = []
        
        ## Deep Learning: Data Loaders
        self.train_loader = DataLoader([])
        self.test_loader = DataLoader([])
        self.num_classes = 0

        # Active Learning: Data Loaders
        self.train_loader = DataLoader([])
        self.pool_unlabelled_loader = DataLoader([])
        self.bucket_labelled_loader = DataLoader([])
        
        ## LOAD DATA LOADERS
        self.configure_data_loaders_main()

        # Configure unlabelled_loader and labelled_loader
        if is_active_learning:
            self.configure_data_loaders_active_learning_random()

        self.logger.warning("==============================================================================\n")

    def configure_data_loaders_main(self):        
        self.logger.warning("------------------------------------\n")
        self.logger.warning("Configuring data loaders for main training...")
        img_size = self.img_size
        data_dir = self.data_dir
        batch_size = self.batch_size
        
        # Transforms
        transform1 = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Aplicar 6 técnicas de aumento de dados
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.8, 1.2))
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

        # Setup data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.num_classes = len(train_dataset.classes)

        self.logger.warning("------------------------------------\n")
        self.logger.warning(f"Train loader size: {len(train_dataset)}")
        self.logger.warning(f"Test loader size: {len(test_dataset)}")
        self.logger.warning(f"Number of classes: {self.num_classes}")
        self.logger.warning(f"Classes: {train_dataset.classes}")

    def configure_data_loaders_active_learning_random(self):
        self.logger.warning("------------------------------------\n")
        self.logger.warning("Configuring data loaders for active learning (random selection)...")
        
        train_dataset = self.train_loader

        # Imprimir o item 0 do train_loader (img, classe, path)
        print(train_dataset.dataset.samples[0])
        print(train_dataset.dataset.classes[train_dataset.dataset.samples[0][1]])
        print(train_dataset.dataset.samples[0][0])
        import matplotlib.pyplot as plt
        import numpy as np
        plt.imshow(np.transpose(train_dataset.dataset.samples[0][0], (1, 2, 0)))
        plt.title(f"Classe: {train_dataset.dataset.classes[train_dataset.dataset.samples[0][1]]}")
        
        plt.show()
        exit()
        
        self.logger.warning(f"Initial labeled set size: {len(labeled_subset)}")
        self.logger.warning(f"Initial unlabeled set size: {len(unlabeled_subset)}")
        self.logger.warning("------------------------------------\n")

    def select_samples_random(self, n_query):
        self.logger.warning("------------------------------------\n")
        self.logger.warning(f"Selecting {n_query} samples randomly from the unlabeled pool...")
        
        # Selecionar aleatoriamente {n_query} índices do pool_unlabelled_loader
        unlabeled_indices = list(range(len(self.pool_unlabelled_loader.dataset)))
        random.seed(self.seed)
        selected_indices = random.sample(unlabeled_indices, min(n_query, len(unlabeled_indices)))
        
        self.logger.warning(f"Selected indices: {selected_indices}")
        self.logger.warning("------------------------------------\n")
        return selected_indices

    def update_datasets(self, samples_selected):
        self.logger.warning("------------------------------------\n")
        self.logger.warning("Updating labeled and unlabeled datasets...")

        unlabeled_dataset = self.pool_unlabelled_loader.dataset
        labeled_dataset = self.bucket_labelled_loader.dataset
        
        # Obter os índices que NÃO foram selecionados para o pool não rotulado
        remaining_indices = list(set(range(len(unlabeled_dataset))) - set(samples_selected))
        
        # Adicionar os índices selecionados ao conjunto rotulado
        new_labeled_subset = Subset(unlabeled_dataset, samples_selected)
        updated_labeled_dataset = labeled_dataset + new_labeled_subset
        
        # Criar um novo subset para os índices restantes no pool não rotulado
        updated_unlabeled_dataset = Subset(unlabeled_dataset, remaining_indices)

        # Atualizar os data loaders
        self.bucket_labelled_loader = DataLoader(updated_labeled_dataset, batch_size=self.batch_size, shuffle=True)
        self.pool_unlabelled_loader = DataLoader(updated_unlabeled_dataset, batch_size=self.batch_size, shuffle=True)

        self.logger.warning(f"Updated labeled set size: {len(self.bucket_labelled_loader.dataset)}")
        self.logger.warning(f"Updated unlabeled set size: {len(self.pool_unlabelled_loader.dataset)}")

        for inputs, labels in self.pool_unlabelled_loader:
                import matplotlib.pyplot as plt
                import numpy as np
                plt.imshow(np.transpose(inputs[0], (1, 2, 0)))
                plt.title(f"Classe: {self.get_name_classes()[labels[0]]}")
                plt.show()
                exit()  
        # Imprimir a frequência de cada classe no pool_unlabelled_loader
        # self.print_class_frequencies(self.pool_unlabelled_loader, "Unlabeled Pool")

        self.logger.warning("------------------------------------\n")


    def get_pool_unlabelled_loader(self):
        return self.pool_unlabelled_loader
    
    def get_bucket_labelled_loader(self):
        return self.bucket_labelled_loader

    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader
    
    def get_num_classes(self):
        return self.num_classes
    
    def get_name_classes(self):
        self.classes = self.train_loader.dataset.classes
        return self.classes
    
    def get_data_dir(self):
        return self.data_dir
    
    @staticmethod
    def load_image_rgb(image_path):
        image = Image.open(image_path).convert("RGB")
        # retornar a imagem e suas dimensões
        return image, image.size
    
    def verify(self):
        data_dir = self.data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"Directory {data_dir} not found.")
        if not os.path.exists(os.path.join(data_dir, "train")):
            raise Exception(f"Directory {os.path.join(data_dir, 'train')} not found.")
        if not os.path.exists(os.path.join(data_dir, "test")):
            raise Exception(f"Directory {os.path.join(data_dir, 'test')} not found.")
        
        # batch_size > 0
        if self.batch_size <= 0:
            raise Exception(f"Batch size must be greater than 0.")
        
        # img_size > 0
        if self.img_size <= 0:
            raise Exception(f"Image size must be greater than 0.")
        
        if self.logger is None:
            raise Exception(f"Logger not found.")
        
        if self.seed is None:
            raise Exception(f"Seed not found.")