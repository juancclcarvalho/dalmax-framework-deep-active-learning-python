from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class DANINHAS_Hander(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        # self.transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x)
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)
    

class CIFAR10_Handler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x)
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)