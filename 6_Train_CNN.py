import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

class CubettiCNNClassifier(nn.Module):

  def __init__(self, num_classes=2):
    super(CubettiCNNClassifier, self).__init__()
    # Caricamento ResNet-18 pre-addestrata
    self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    num_ftrs = self.resnet.fc.in_features
    self.resnet.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes), 
    )

  def forward(self, x):
    return self.resnet(x)

# ---------------------------
# 1. ESECUZIONE DEL TRAINING  
# ---------------------------
if __name__ == '__main__':
  EXCEL_PATH = 'Dataset_cubetti.xlsx'
  IMAGE_DIR = 'Immagini_cubetti/'
  BENCHMARK_FILE = 'Riferimento.jpg'  # File di riferimento FFF

  transform_pipeline = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  print("Inizializzazione del modello CNN (ResNet-18 Transfer Learning)...")
  model = CubettiCNNClassifier(num_classes=2)

  # -------------------------------------
  # 2. CONTROLLO DISPONIBILITà GPU O CPU
  # -------------------------------------
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  print(f"Modello spostato con successo sul dispositivo: {device}")

  print(
      "\nLa CNN è configurata e pronta per l'addestramento sui cubetti!"
  )