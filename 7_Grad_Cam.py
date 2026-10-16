import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F
from torchvision import models, transforms


class GradCAM:

  def __init__(self, model, target_layer):
    self.model = model
    self.target_layer = target_layer
    self.gradients = None
    self.target_layer.register_forward_hook(self.save_activation)
    self.target_layer.register_full_backward_hook(self.save_gradient)

  def save_activation(self, module, input, output):
    self.activation = output.detach()

  def save_gradient(self, module, grad_input, grad_output):
    self.gradients = grad_output[0].detach()

  def __call__(self, x, class_idx=None):
    self.model.eval()
    output = self.model(x)
    if class_idx is None:
      class_idx = output.argmax(dim=1).item()
    self.model.zero_grad()
    score = output[0, class_idx]
    score.backward()

    gradients = self.gradients[0]
    activation = self.activation[0]
    weights = torch.mean(gradients, dim=(1, 2))

    cam = torch.zeros(activation.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
      cam += w * activation[i]

    cam = F.relu(cam)
    cam = cam - cam.min()
    if cam.max() > 0:
      cam = cam / cam.max()
    return cam.cpu().numpy()

# ------------------------------------------
# 1. ESECUZIONE GRAD-CAM + RIFERIMENTO SSIM
# ------------------------------------------
if __name__ == '__main__':
  IMAGE_DIR = 'Immagini_cubetti/'

  BENCHMARK_FILE = 'Riferimento.jpg'

  # Caricamento dell'immagine di benchmark per il confronto
  bench_path = os.path.join(IMAGE_DIR, BENCHMARK_FILE)
  bench_raw = cv2.imread(bench_path)
  if bench_raw is None:
    raise FileNotFoundError(
        f"Impossibile trovare l'immagine di benchmark: {bench_path}"
    )

  bench_rgb = cv2.cvtColor(bench_raw, cv2.COLOR_BGR2RGB)
  bench_resized = cv2.resize(bench_rgb, (224, 224))
  gray_bench = cv2.cvtColor(bench_resized, cv2.COLOR_RGB2GRAY)

  # Lista dei 10 cubetti richiesti per l'analisi
  target_cubes = [
      'a1.jpg',
      'd1.jpg',
      'g2.jpg',
      'h2.jpg',
      'i3.jpg',
      'q3.jpg',
      'r3.jpg',
      'v1.jpg',
      'y2.jpg',
      'z3.jpg',
  ]

  # Caricamento del modello ResNet-18 e configurazione Grad-CAM
  model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
  model.eval()
  target_layer = model.layer4[1].conv2
  grad_cam = GradCAM(model, target_layer)

  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  results = []

  print(
      'Elaborazione in corso: calcolo SSIM rispetto al riferimento e generazione Grad-CAM'
  )
  for cube_name in target_cubes:
    img_path = os.path.join(IMAGE_DIR, cube_name)
    raw_img = cv2.imread(img_path)
    if raw_img is None:
      print(f"[ATTENZIONE] Impossibile trovare l'immagine: {img_path}")
      continue

    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    # ----------------------------------------------------
    # 2. CALCOLO DELL'INDICE SSIM RISPETTO AL RIFERIMENTO
    # ----------------------------------------------------
    gray_sample = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    ssim_score, _ = ssim(gray_bench, gray_sample, full=True)

    # --------------------------------------
    # 3. GENERAZIONE DELLA HEATMAP Grad-CAM
    # --------------------------------------
    input_tensor = transform(img_resized).unsqueeze(0)
    heatmap = grad_cam(input_tensor)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed_img = np.uint8(0.5 * img_resized + 0.5 * heatmap_colored)

    results.append((cube_name, img_resized, superimposed_img, ssim_score))

  # Divisione in due gruppi da 5 cubetti ciascuno
  batch1 = results[:5]  # Primi 5 cubetti
  batch2 = results[5:]  # Ultimi 5 cubetti

  # SCHEDA 1: Primi 5 cubetti (Griglia 2x5 con SSIM)
  if batch1:
    fig1, axes1 = plt.subplots(2, len(batch1), figsize=(16, 7))
    for col, (name, orig_img, cam_img, score) in enumerate(batch1):
      # Riga Superiore: Immagine Originale + Punteggio SSIM nel titolo
      axes1[0, col].imshow(orig_img)
      axes1[0, col].set_title(
          f'{name}\nSSIM: {score:.3f}', fontsize=11, fontweight='bold'
      )
      axes1[0, col].axis('off')

      # Riga Inferiore: Immagine con Grad-CAM
      axes1[1, col].imshow(cam_img)
      axes1[1, col].set_title('Grad-CAM', fontsize=10, color='darkred')
      axes1[1, col].axis('off')

    fig1.suptitle(
        'Analisi qualitativa - Gruppo 1 (Riferimento SSIM vs Grad-CAM)',
        fontsize=14,
        fontweight='bold',
    )
    plt.tight_layout()

  # SCHEDA 2: Ultimi 5 cubetti (Griglia 2x5 con SSIM)
  if batch2:
    fig2, axes2 = plt.subplots(2, len(batch2), figsize=(16, 7))
    for col, (name, orig_img, cam_img, score) in enumerate(batch2):
      # Riga Superiore: Immagine Originale + Punteggio SSIM nel titolo
      axes2[0, col].imshow(orig_img)
      axes2[0, col].set_title(
          f'{name}\nSSIM: {score:.3f}', fontsize=11, fontweight='bold'
      )
      axes2[0, col].axis('off')

      # Riga Inferiore: Immagine con Grad-CAM
      axes2[1, col].imshow(cam_img)
      axes2[1, col].set_title('Grad-CAM', fontsize=10, color='darkred')
      axes2[1, col].axis('off')

    fig2.suptitle(
        'Analisi qualitativa - Gruppo 2 (Riferimento SSIM vs Grad-CAM)',
        fontsize=14,
        fontweight='bold',
    )
    plt.tight_layout()

  print(
      '\nElaborazione completata con successo! Apertura delle finestre riassuntive.'
  )
  plt.show()