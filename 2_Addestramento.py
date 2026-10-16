import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ----------------------------------------
# 1. CARICAMENTO E PREPROCESSING DEI DATI
# ----------------------------------------
df = pd.read_csv('Dataset_cubetti.csv')

# Nomi delle colonne di INPUT (parametri di ingresso)
X_cols = ['Temperatura', 'Distanza_Linea', 'Flow_Rate']

# Nomi delle colonne di OUTPUT che si vogliono predire 
y_cols = ['Scostamento_Max', 'Scostamento_Med', 'Dist_01', 'Dev_Standard', 'Massa']

X = df[X_cols].values
y = df[y_cols].values

# Split Train (80%) / Test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardizzazione delle variabili (media 0, deviazione standard 1)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Si salvano gli scaler per riutilizzarli
with open('scalers.pkl', 'wb') as f:
    pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)

# Conversione in tensori PyTorch
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# -----------------------------------------------
# 2. DEFINIZIONE DELL'ARCHITETTURA NEURALE (MLP)
# -----------------------------------------------
class QualityMLP(nn.Module):
    def __init__(self, input_dim=3, output_dim=4):
        super(QualityMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

model = QualityMLP(input_dim=len(X_cols), output_dim=len(y_cols))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# --------------------------
# 3. CICLO DI ADDESTRAMENTO
# --------------------------
epochs = 200
loss_history = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    loss_history.append(epoch_loss)
    
    if (epoch + 1) % 40 == 0:
        print(f"Epoca [{epoch+1}/{epochs}] - Loss (MSE): {epoch_loss:.4f}")

# Salvataggio del modello addestrato
torch.save(model.state_dict(), 'mlp_quality_model.pth')
print("\n Modello salvato come 'mlp_quality_model.pth'")

# -------------------------------------------
# 4. VALUTAZIONE E VERIFICA DELLE PREDIZIONI
# -------------------------------------------
model.eval()
with torch.no_grad():
    predictions_scaled = model(X_test_t).numpy()

# Riportiamo le predizioni nella scala reale (mm e grammi)
predictions_real = scaler_y.inverse_transform(predictions_scaled)
y_test_real = scaler_y.inverse_transform(y_test_scaled)

print("\n--- Confronto tra Valori Reali e Predetti (Primo campione di Test) ---")
for i, col in enumerate(y_cols):
    reale = y_test_real[0][i]
    predetto = predictions_real[0][i]
    print(f"{col:20s} | Reale: {reale:7.3f} | Predetto: {predetto:7.3f}")

# Grafico della Loss dell'addestramento
plt.figure(figsize=(7, 4))
plt.plot(loss_history, label='Loss MSE')
plt.xlabel('Epoche')
plt.ylabel('Errore Medio Quadratico (MSE)')
plt.title('Andamento dell\'addestramento MLP')
plt.legend()
plt.grid(True)
plt.savefig('grafico_loss_mlp.png')
plt.show()