import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CubettiVisionDataset(Dataset):

  def __init__(
      self, excel_path, image_dir, benchmark_filename, transform=None
  ):
    """Dataset personalizzato per integrare i parametri tabulari, le immagini dei cubetti robotizzati e il benchmark di riferimento."""
    
    self.df = pd.read_excel(excel_path)
    self.image_dir = image_dir
    self.transform = transform

    # --------------------------------------------------------------
    # 1. CARICAMENTO DELL'IMMAGINE DEL RIFERIMENTO FFF TRADIZIONALE
    # --------------------------------------------------------------
    bench_path = os.path.join(image_dir, benchmark_filename)
    bench_img = cv2.imread(bench_path)
    if bench_img is None:
      raise ValueError(f"Impossibile trovare l'immagine di benchmark: {bench_path}")

    # ----------------------------------------------------------------------
    # 2. PREPROCESSING DEL RIFERIMENTO (CONVERSIONE RGB E RESIZE A 224x224)
    # ----------------------------------------------------------------------
    self.benchmark = cv2.resize(
        cv2.cvtColor(bench_img, cv2.COLOR_BGR2RGB), (224, 224)
    )

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    # Estrazione del nome del file immagine dalla riga del dataset (es. "a1.jpg")
    img_name = self.df.loc[idx, 'File_Imm']
    img_path = os.path.join(self.image_dir, img_name)

    # ------------------------------------
    # 3. CARICAMENTO IMMAGINE CUBETTO RAM
    # ------------------------------------
    image = cv2.imread(img_path)
    if image is None:
      raise FileNotFoundError(f"Impossibile trovare l'immagine: {img_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    # -----------------------------------------------------------------
    # 4. CALCOLO METRICHE DI CONFRONTO E CONVERSIONE IN SCALA DI GRIGI
    # -----------------------------------------------------------------
    gray_sample = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_bench = cv2.cvtColor(self.benchmark, cv2.COLOR_RGB2GRAY)
    similarity_score, _ = ssim(gray_bench, gray_sample, full=True)

    # ----------------------------------------
    # 5. TRASFORMAZIONI PyTorch SULL'IMMAGINE
    # ----------------------------------------
    if self.transform:
      image_tensor = self.transform(image)
    else:
      # Conversione manuale in tensore PyTorch [C, H, W] e normalizzazione [0, 1]
      image_tensor = (
          torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
      )

    sample = {
        'image': image_tensor,
        'similarity_ssim': torch.tensor(
            similarity_score, dtype=torch.float32
        ),  # Metrica visiva
        'file_name': img_name,
    }

    return sample

# -----------------------
# 6. ESECUZIONE DEL TEST 
# -----------------------
if __name__ == '__main__':
  EXCEL_PATH = 'Dataset_cubetti.xlsx'
  IMAGE_DIR = 'Immagini_cubetti/'

  BENCHMARK_FILE = 'Riferimento.jpg'

  # Trasformazioni standard per reti neurali
  transform_pipeline = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  print('Inizializzazione del Dataset PyTorch con controllo Riferimento...')

  print(
      'Struttura del dataset pronta per CNN.'
  )