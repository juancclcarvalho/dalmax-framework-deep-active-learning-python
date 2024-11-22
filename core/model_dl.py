import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import time
import enum

import os
import torch
from torchvision import transforms
from PIL import Image

import torch.backends.cudnn as cudnn

class ModelType(enum.Enum):
    RESNET18 = 1
    RESNET50 = 2
    RESNET101 = 3

class ModelResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ModelResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
class ModelResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ModelResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
class ModelResNet101(nn.Module):
    def __init__(self, num_classes):
        super(ModelResNet101, self).__init__()
        self.model = models.resnet101(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class DeepLearning():
    def __init__(self, model_type: ModelType, num_classes: int, learning_rate: float=0.001, logger: object=None):
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.model_type = model_type
        self.num_classes = num_classes

        self.create_model()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    # Function to create the model
    def create_model(self):
        self.logger.warning(f"Creating model...")
        self.logger.warning(f"Model: {self.model_type.name}")
        self.logger.warning(f"Number of classes: {self.num_classes}")
        self.logger.warning(f"Using device: {self.device}")

        if self.model_type == ModelType.RESNET18:
            self.model = ModelResNet18(self.num_classes).to(self.device)
        elif self.model_type == ModelType.RESNET50:
            self.model = ModelResNet50(self.num_classes).to(self.device)
        elif self.model_type == ModelType.RESNET101:
            self.model =  ModelResNet101(self.num_classes).to(self.device)
        else:
            raise ValueError("Model not found.")
        
        if self.device.__str__() == 'cuda':
            self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True
            self.logger.warning("Using {} GPUs!".format(torch.cuda.device_count()))

    # Fuction to train the model
    def train_model(self, train_loader, num_epochs):
        train_loss = []
        train_acc = []
        all_times = []
        print("Device: ", self.device)

        self.logger.warning("Init training...")

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            start_time = time.time()

            for inputs, labels in train_loader:
                print("Inputs: ", inputs)
                print("Labels: ", labels)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zera os gradientes
                self.optimizer.zero_grad()

                # Forward
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Atualiza métricas
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / total
            epoch_acc = correct / total

            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)

            end_time = time.time()
            total_time_in_seconds = end_time - start_time
            all_times.append(total_time_in_seconds)

            self.logger.warning(f"Epoch {epoch+1}/{num_epochs} - Time: {total_time_in_seconds:.2f}s - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
        
        self.logger.warning("Treinamento finalizado em {:.2f}s.".format(sum(all_times)))
        return train_loss, train_acc

    # Function to evaluate the model
    def evaluate_model(self, test_loader):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        self.logger.warning("Evaluating model on test data...")

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = test_loss / total
        accuracy = correct / total

        self.logger.warning(f"Test Loss: {avg_loss:.4f} - Test Accuracy: {accuracy:.4f}")

        return avg_loss, accuracy

    # Function to save the model
    def save_model(self, path, dir_results):
        self.logger.warning("Saving model...")
        path = dir_results + "/" + path
        torch.save(self.model, path)
        self.logger.warning(f"Model saved in '{path}'.")

    # Function to plot loss and accuracy curves
    def plot_loss_accuracy(self, train_loss, train_acc, num_epochs, data_results):
        self.logger.warning("Plotting loss and accuracy curves...")
        # Plota as curvas de treinamento
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs + 1), train_loss, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss durante o treinamento")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs + 1), train_acc, label="Train Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Acurácia durante o treinamento")
        plt.legend()

        plt.tight_layout()
        
        # Salvar em um arquivo .pdf
        plt.savefig(data_results + "/training_curves_loss_accuracy.pdf")
        
        self.logger.warning(f"Training curves saved in '{data_results}/training_curves_loss_accuracy.pdf'.")

    # Function to load a model
    def load_model_pth(self, model_path):
        self.model = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval() 
        return self.model

    # Fuction to predict an image
    def predict_image(self, image_path, transform=None, classes=None, img_size=128):
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),  # Dimensiona para o formato esperado pelo modelo
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        image = Image.open(image_path).convert("RGB")

        image_tensor = transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)  # Obtém as probabilidades
            predicted_idx = torch.argmax(probabilities, dim=1).item()  # Índice da classe com maior probabilidade
            confidence = probabilities[0, predicted_idx].item()  # Confiança da predição
        
        predicted_class = classes[predicted_idx]
        return predicted_class, confidence