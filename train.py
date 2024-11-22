import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import time

import torch.backends.cudnn as cudnn
import logging
import time
from PIL import Image

# Configuração do logger
logger = logging.getLogger(__name__)  # Define o logger apenas para o seu módulo
logger.setLevel(logging.DEBUG)

# Criar um handler para escrever no arquivo de log
text_time_log = time.strftime('%Y-%m-%d-%H-%M-%S')
PATH_LOG_FINAL = 'logs/' + text_time_log + '-log-dalmax.log'
file_handler = logging.FileHandler(PATH_LOG_FINAL)
file_handler.setLevel(logging.DEBUG)

# Criar um handler para imprimir no console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Definir o formato do log
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Adicionar os handlers ao logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Configurações básicas
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.warning(f"Usando dispositivo: {device}")


# Função para carregar os dados
def load_data(data_dir, batch_size, img_size):
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

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Tamanho compatível com ResNet-50
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, len(train_dataset.classes)

# Modelo ResNet-50 com pesos pré-treinados
def create_model(num_classes):
    model = models.resnet50(pretrained=True)
    # Substitui a última camada totalmente conectada para o número de classes do dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model.to(device)

# Função de treinamento
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    train_loss = []
    train_acc = []
    all_times = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zera os gradientes
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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

        logger.warning(f"Epoch {epoch+1}/{num_epochs} - Time: {total_time_in_seconds:.2f}s - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
    
    logger.warning("Treinamento finalizado em {:.2f}s.".format(sum(all_times)))
    return train_loss, train_acc

# Função de avaliação
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / total
    accuracy = correct / total

    return avg_loss, accuracy

"""
Retorna 2 parametros: label da classe predita e a acurácia
"""
def predict_image(image_path, model, transform, classes):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        prob = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

    return classes[predicted.item()], prob[predicted.item()].item()

def load_model_pth(model_path, num_classes):
    model = create_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    return model

# Função principal
def main(data_dir, batch_size=32, img_size=224, num_epochs=100, learning_rate=0.001):
    # Print infos do treinamento
    logger.warning(f"Configurações: batch_size={batch_size}, img_size={img_size}, num_epochs={num_epochs}, learning_rate={learning_rate}")
    # Carrega os dados
    train_loader, test_loader, num_classes = load_data(data_dir, batch_size, img_size)

    # Imprmir a primeira imagem do dataset train_loader: primir o nome do arquivo e a classe e mostrar a imagem
    logger.warning("Primeira imagem do dataset:")
    logger.warning(f"Nome do arquivo: {train_loader.dataset.samples[2][0]}")
    logger.warning(f"Classe: {train_loader.dataset.classes[train_loader.dataset.samples[2][1]]}")
    """
    plt.imshow(train_loader.dataset[2][0].permute(1, 2, 0))
    plt.axis("off")
    plt.show()"""


    # Print classes
    logger.warning(f"Classes: {train_loader.dataset.classes}")
    logger.warning(f"Número de classes: {num_classes}")

    # Pegar um batch de dados com exatamente 10 imagens de cada classe
    # Pegar um batch de dados com exatamente 10 imagens de cada classe
    
    class_counts = {cls: 0 for cls in range(num_classes)}
    selected_images = []
    selected_labels = []

    for inputs, labels in train_loader:
        for input, label in zip(inputs, labels):
            if class_counts[label.item()] < 10:
                selected_images.append(input)
                selected_labels.append(label)
                class_counts[label.item()] += 1
            if all(count >= 10 for count in class_counts.values()):
                break
        if all(count >= 10 for count in class_counts.values()):
            break

    selected_images = torch.stack(selected_images)
    selected_labels = torch.stack(selected_labels)

    INDEX = selected_labels[0].item()
    logger.warning(f"Nome do arquivo Classe selecionada: {train_loader.dataset.samples[INDEX][0]}")


    # Imprimir o nome original de cada imagem selecionada
    logger.warning("Imagens selecionadas:")
    for i, label in enumerate(selected_labels):
        # logger.warning(f"Imagem {i+1}: {train_loader.dataset.samples[label.item()][0]}")
        pass

    # Show 5 image grid and labels
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(selected_images[i].permute(1, 2, 0))
        plt.title(train_loader.dataset.classes[selected_labels[i].item()])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("selected_images.pdf")

    # Imprimir a porcentagem de cada classe selecionada selected_images
    logger.warning("Porcentagem de cada classe selecionada:")
    for cls, count in class_counts.items():
        logger.warning(f"Classe {cls}: {count} ({count / 10 * 100:.0f}%)")

    # Cria o modelo
    model = create_model(num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.__str__() == 'cuda':
        logger.warning("Multiplos GPUs disponíveis.")
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
        logger.warning("Usando {} GPUs.".format(torch.cuda.device_count()))

    # Define a função de perda e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Treinamento
    logger.warning("Iniciando treinamento...")
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, num_epochs)

    # Avaliação
    logger.warning("Avaliando no conjunto de testes...")
    test_loss, test_acc = evaluate_model(model, test_loader, criterion)

    logger.warning(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_acc:.4f}")

    # Salva o modelo treinado
    torch.save(model, "model_final.pth")
    logger.warning("Modelo salvo em 'model_final.pth'.")

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
    plt.savefig("training_curves.pdf")

if __name__ == "__main__":
    # Substitua 'Pasta/' pelo caminho real do seu dataset
    main(data_dir="DATA/daninhas_full", batch_size=256, img_size=128, num_epochs=10, learning_rate=0.001)
