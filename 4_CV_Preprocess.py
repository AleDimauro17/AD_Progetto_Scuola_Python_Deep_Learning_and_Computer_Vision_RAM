import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------
# 1. CONFIGURAZIONE DEI PERCORSI
# -------------------------------
EXCEL_PATH = 'Dataset_cubetti.xlsx'  
IMAGE_DIR = 'Immagini_cubetti/' 

# ---------------------------
# 2. CARICAMENTO DEL DATASET
# ---------------------------
print("Caricamento del dataset Excel...")
df = pd.read_excel(EXCEL_PATH)

# Verifica rapida delle prime righe e della colonna File_Imm
print(f"Dataset caricato con successo! Trovate {len(df)} righe.")
print(df[['File_Imm']].head())

# -----------------------------------------
# 3. PARAMETRI DI PREPROCESSING PER LA CNN
# -----------------------------------------
TARGET_SIZE = (224, 224)  # Risoluzione standard

# Funzione di preprocessing di un'immagine
def preprocess_image(file_name, image_dir, target_size):
  img_path = os.path.join(image_dir, file_name)

  # Caricamento dell'immagine con OpenCV
  img = cv2.imread(img_path)

  if img is None:
    print(
        f"[ATTENZIONE] Impossibile trovare o caricare l'immagine: {img_path}"
    )
    return None

  # Conversione in RGB per una corretta visualizzazione
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # Ridimensionamento alla dimensione fissa per la CNN
  img_resized = cv2.resize(img_rgb, target_size)

  # Normalizzazione dei pixel nell'intervallo [0, 1] 
  img_normalized = img_resized / 255.0

  return img_normalized

# ------------------------------------------------------
# 4. TEST DI CARICAMENTO SUL PRIMO CUBETTO (ex. a1.jpg)
# ------------------------------------------------------
print("\nTest di preprocessing sul primo campione...")
primo_file = df.loc[0, 'File_Imm'] 
processed_img = preprocess_image(primo_file, IMAGE_DIR, TARGET_SIZE)

if processed_img is not None:
  print(
      f"Immagine '{primo_file}' caricata e preprocessata correttamente! Shape"
      f" risultante: {processed_img.shape}"
  )

  # Visualizzazione di confronto tramite Matplotlib
  plt.figure(figsize=(5, 5))
  plt.imshow(processed_img)
  plt.title(f"Preprocessing completato: {primo_file}")
  plt.axis('off')
  plt.show()