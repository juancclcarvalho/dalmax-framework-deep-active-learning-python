import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image
import random
from torch.utils.data import DataLoader, Subset

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
        
        # Dataset completo
        full_dataset = self.train_loader.dataset
        num_classes = len(full_dataset.classes)
        total_images = self.n_init_labeled

        if total_images > len(full_dataset):
            raise ValueError("n_init_labeled is greater than the total number of images in the dataset.")

        # Selecionar índices de forma aleatória
        labelled_indices = random.sample(range(len(full_dataset)), total_images)

        # Criar o conjunto rotulado
        labelled_dataset = Subset(full_dataset, labelled_indices)
        self.bucket_labelled_loader = DataLoader(labelled_dataset, batch_size=self.batch_size, shuffle=False)

        # Calcular a frequência de cada classe no bucket_labelled_loader
        class_counts = {class_idx: 0 for class_idx in range(num_classes)}
        for idx in labelled_indices:
            label = full_dataset.targets[idx]
            class_counts[label] += 1

        self.logger.warning("------------------------------------\n")
        self.logger.warning("Before configuring pool and bucket loaders:")
        self.logger.warning(f"Pool unlabelled loader size: {len(full_dataset)}")
        self.logger.warning(f"Bucket labelled loader size: {len(labelled_dataset)}")

        self.logger.warning("\nClass frequencies in bucket_labelled_loader:")
        for class_idx, count in class_counts.items():
            class_name = full_dataset.classes[class_idx]
            self.logger.warning(f"Class '{class_name}' contains {count} images")
        
        # Criar o conjunto não rotulado
        unlabelled_indices = list(set(range(len(full_dataset))) - set(labelled_indices))
        unlabelled_dataset = Subset(full_dataset, unlabelled_indices)

        self.pool_unlabelled_loader = unlabelled_dataset
        self.bucket_labelled_loader = labelled_dataset

        self.logger.warning("------------------------------------\n")
        self.logger.warning("After configuring pool and bucket loaders:")
        print("Pool unlabelled loader size: ", len(self.pool_unlabelled_loader))
        print("Bucket labelled loader size: ", len(self.bucket_labelled_loader))

        # Classes 
        print("Classes: ", self.pool_unlabelled_loader.dataset.classes)

    def configure_data_loaders_active_learning_init_balanced(self):
        self.logger.warning("------------------------------------\n")
        self.logger.warning("Configuring data loaders for active learning...")
        # Dataset completo
        full_dataset = self.train_loader.dataset
        num_classes = len(full_dataset.classes)
        images_per_class = self.n_init_labeled // num_classes

        self.logger.warning("------------------------------------\n")
        self.logger.warning("Before configuring pool and bucket loaders:")
        self.logger.warning(f"Pool unlabelled loader size: {len(full_dataset)}")
        self.logger.warning(f"Bucket labelled loader size: {len(self.bucket_labelled_loader)}")

        if images_per_class == 0:
            raise ValueError("n_init_labeled is too small to distribute across all classes.")

        # Organizar índices por classe
        class_to_indices = {class_idx: [] for class_idx in range(num_classes)}
        for idx, (_, label) in enumerate(full_dataset.samples):
            class_to_indices[label].append(idx)

        # Garantir que há imagens suficientes para a distribuição
        for class_idx, indices in class_to_indices.items():
            if len(indices) < images_per_class:
                raise ValueError(f"Not enough images in class '{full_dataset.classes[class_idx]}' for the initial labeled pool.")

        # Selecionar imagens para o conjunto rotulado
        labelled_indices = []
        for class_idx in range(num_classes):
            labelled_indices.extend(class_to_indices[class_idx][:images_per_class])

        # Criar o conjunto rotulado
        labelled_dataset = Subset(full_dataset, labelled_indices)
        self.bucket_labelled_loader = DataLoader(labelled_dataset, batch_size=self.batch_size, shuffle=False)

        # Calcular a frequência de cada classe no bucket_labelled_loader
        class_counts = {class_idx: 0 for class_idx in range(num_classes)}
        for idx in labelled_indices:
            label = full_dataset.targets[idx]
            class_counts[label] += 1

        self.logger.warning("Class frequencies in bucket_labelled_loader:")
        for class_idx, count in class_counts.items():
            class_name = full_dataset.classes[class_idx]
            self.logger.warning(f"Class '{class_name}' contains {count} images")
        # Criar o conjunto não rotulado
        unlabelled_indices = list(set(range(len(full_dataset))) - set(labelled_indices))
        unlabelled_dataset = Subset(full_dataset, unlabelled_indices)
        self.pool_unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.logger.warning("------------------------------------\n")
        self.logger.warning("After configuring pool and bucket loaders:")
        self.logger.warning(f"Pool unlabelled loader size: {len(unlabelled_dataset)}")
        self.logger.warning(f"Bucket labelled loader size: {len(labelled_dataset)}")

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
