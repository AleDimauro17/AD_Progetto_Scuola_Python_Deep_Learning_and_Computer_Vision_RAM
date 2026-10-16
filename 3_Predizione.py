import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

# ---------------------------------------------------
# 1. ARCHITETTURA RETE NEURALE E CARICAMENTO MODELLI
# ---------------------------------------------------
class QualityMLP(nn.Module):
    def __init__(self, input_dim=3, output_dim=5):
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

# Caricamento degli scaler salvati
with open('scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)
    scaler_X = scalers['scaler_X']
    scaler_y = scalers['scaler_y']

# Caricamento del modello PyTorch
model = QualityMLP(input_dim=3, output_dim=5)
model.load_state_dict(torch.load('mlp_quality_model.pth'))
model.eval()

# -------------------------------------------------
# 2. CARICAMENTO DATASET E IDENTIFICAZIONE COLONNE
# -------------------------------------------------
try:
    df = pd.read_csv('Dataset_cubetti.csv', sep=None, engine='python')
except Exception:
    df = pd.read_csv('Dataset_cubetti.csv')

df.columns = df.columns.str.strip().str.lower()

X_cols = []
for keyword in ['temp', 'dist', 'flow']:
    col_found = [c for c in df.columns if keyword in c]
    if col_found:
        X_cols.append(col_found[0])

if len(X_cols) < 3:
    colonne_valide = [c for c in df.columns if 'id' not in c and 'file' not in c]
    X_cols = colonne_valide[:3]

y_cols = [c for c in df.columns if c not in X_cols and 'id' not in c and 'file' not in c]

# Individuazione indici per Massa e Scostamento Massimo
scost_max_idx = 0
massa_col_idx = 4
for idx, col in enumerate(y_cols):
    if 'max' in col:
        scost_max_idx = idx
    if 'massa' in col:
        massa_col_idx = idx

X = df[X_cols].values
y = df[y_cols].values

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_scaled = scaler_X.transform(X_test)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy()

y_pred_real = scaler_y.inverse_transform(y_pred_scaled)

# -----------------------------------------------
# 3. FUNZIONI DI PREDIZIONE E CALCOLO GRIGLIE 3D
# -----------------------------------------------
def calcola_predizione(temp, dist, flow):
    input_data = np.array([[temp, dist, flow]])
    input_scaled = scaler_X.transform(input_data)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = model(input_tensor).numpy()
    return scaler_y.inverse_transform(pred_scaled)[0]

def eval_semaforo(massa):
    # Riferimento FFF = 5.0 g
    if 4.5 <= massa <= 5.5:
        return '#2ecc71', '🟢 CONFORME'
    elif (3.5 <= massa < 4.5) or (5.5 < massa <= 6.5):
        return '#f39c12', '🟡 TOLLERANZA MEDIA'
    else:
        return '#e74c3c', '🔴 FUORI TOLLERANZA'

# Valutazione griglia vettorializzata per superfici 3D
def calcola_griglia_3d(x1_vals, x2_vals, val_fisso, tipo_fisso='dist'):
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    N = X1.size
    x1_flat = X1.ravel()
    x2_flat = X2.ravel()
    
    if tipo_fisso == 'dist':
        pts = np.column_stack([x1_flat, np.full(N, val_fisso), x2_flat])
    elif tipo_fisso == 'flow':
        pts = np.column_stack([x1_flat, x2_flat, np.full(N, val_fisso)])
    else:  # tipo_fisso == 'temp'
        pts = np.column_stack([np.full(N, val_fisso), x1_flat, x2_flat])
        
    pts_scaled = scaler_X.transform(pts)
    pts_tensor = torch.tensor(pts_scaled, dtype=torch.float32)
    with torch.no_grad():
        preds_scaled = model(pts_tensor).numpy()
    preds_real = scaler_y.inverse_transform(preds_scaled)
    
    Z_scost = preds_real[:, scost_max_idx].reshape(X1.shape)
    Z_massa = preds_real[:, massa_col_idx].reshape(X1.shape)
    return X1, X2, Z_scost, Z_massa

# ----------------------
# 4. DEFINIZIONE LAYOUT 
# ----------------------
fig = plt.figure(figsize=(18, 10))
fig.suptitle('Dashboard Interattiva', fontsize=14, fontweight='bold')

# Griglia: 3 righe (Alto: Grafici 2D, Centro: Box Semaforo, Basso: Grafici 3D)
gs = GridSpec(3, 12, figure=fig, height_ratios=[1.0, 0.35, 1.0], hspace=0.50, wspace=0.40)

# --- 1. IN ALTO: Grafici 2D ---
ax_bars = fig.add_subplot(gs[0, 0:5])
ax_scatter = fig.add_subplot(gs[0, 7:12])

# --- 2. AL CENTRO: Box Semaforo ---
ax_status = fig.add_subplot(gs[1, 3:9])

# --- 3. IN BASSO: 4 Grafici 3D ---
ax_3d_1 = fig.add_subplot(gs[2, 0:3], projection='3d')
ax_3d_2 = fig.add_subplot(gs[2, 3:6], projection='3d')
ax_3d_3 = fig.add_subplot(gs[2, 6:9], projection='3d')
ax_3d_4 = fig.add_subplot(gs[2, 9:12], projection='3d')

plt.subplots_adjust(left=0.05, right=0.95, top=0.91, bottom=0.16)

# -----------------------
# 5. POPOLAMENTO GRAFICI
# -----------------------
# Parametri Iniziali
temp_init, dist_init, flow_init = 210.0, 0.60, 100.0
valori_init = calcola_predizione(temp_init, dist_init, flow_init)
massa_init = valori_init[massa_col_idx]
scost_max_init = valori_init[scost_max_idx]

# 1. ISTOGRAMMA PREDIZIONI
labels = ['Scost. Max\n[mm]', 'Scost. Med\n[mm]', 'Dist_01\n[mm]', 'Dev. Std\n[mm]', 'Massa\n[g]']
bars = ax_bars.bar(labels, valori_init, color='#7f8c8d', width=0.5)
ax_bars.set_ylim(0, max(valori_init) * 1.25 if max(valori_init) > 0 else 10)
ax_bars.set_title('Predizioni modello MLP', fontweight='bold', fontsize=11)
ax_bars.grid(axis='y', linestyle='--', alpha=0.6)

text_labels = []
for bar, val in zip(bars, valori_init):
    txt = ax_bars.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.02, 
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    text_labels.append(txt)

# 2. REALE VS PREDETTO
massa_reale_test = y_test[:, massa_col_idx]
massa_pred_test = y_pred_real[:, massa_col_idx]
ax_scatter.scatter(massa_reale_test, massa_pred_test, color='#2c3e50', alpha=0.8, edgecolors='k', label='Campioni set test')
min_m = min(massa_reale_test.min(), massa_pred_test.min()) - 0.1
max_m = max(massa_reale_test.max(), massa_pred_test.max()) + 0.1
ax_scatter.plot([min_m, max_m], [min_m, max_m], 'r--', label='Predizione esatta (y=x)')
ax_scatter.set_title('Confronto massa: reale vs predetto', fontweight='bold', fontsize=11)
ax_scatter.set_xlabel('Valori reali [g]', fontsize=9)
ax_scatter.set_ylabel('Valori predetti [g]', fontsize=9)
ax_scatter.grid(True, linestyle='--', alpha=0.5)
ax_scatter.legend(fontsize=8)

# 3. BOX  
ax_status.axis('off')
color_sem, msg_sem = eval_semaforo(massa_init)
status_box = ax_status.text(
    0.5, 0.5, 
    f"Riferimento FFF: 5.0 g\n\nMassa stimata: {massa_init:.3f} g",
    ha='center', va='center', fontsize=11, fontweight='bold', color='white',
    bbox=dict(boxstyle='round,pad=0.8', facecolor=color_sem, alpha=0.9)
)

# 4. SUPERFICI 3D
T_range = np.linspace(200, 220, 15)
D_range = np.linspace(0.5, 0.7, 15)
F_range = np.linspace(80, 120, 15)

# Griglia 1: Temperatura vs Flow_Rate -> Scostamento massimo (Distanza fissa)
T1, F1, Z_scost_1, _ = calcola_griglia_3d(T_range, F_range, dist_init, 'dist')
surf1 = [ax_3d_1.plot_surface(T1, F1, Z_scost_1, cmap='viridis', alpha=0.75, edgecolor='none')]
pt1 = ax_3d_1.scatter([temp_init], [flow_init], [scost_max_init], color='red', s=60, zorder=5)
ax_3d_1.set_xlabel('T (°C)', fontsize=7)
ax_3d_1.set_ylabel('FR (%)', fontsize=7)
ax_3d_1.set_zlabel('S. Max (mm)', fontsize=7)
ax_3d_1.set_title('A. Temperatura vs Flow Rate -> Scost. Max', fontsize=9, fontweight='bold')

# Griglia 2: Temperatura vs Distanza -> Scostamento massimo (Flow_Rate fisso)
T2, D2, Z_scost_2, _ = calcola_griglia_3d(T_range, D_range, flow_init, 'flow')
surf2 = [ax_3d_2.plot_surface(T2, D2, Z_scost_2, cmap='plasma', alpha=0.75, edgecolor='none')]
pt2 = ax_3d_2.scatter([temp_init], [dist_init], [scost_max_init], color='red', s=60, zorder=5)
ax_3d_2.set_xlabel('T (°C)', fontsize=7)
ax_3d_2.set_ylabel('Dist. (mm)', fontsize=7)
ax_3d_2.set_zlabel('S. Max (mm)', fontsize=7)
ax_3d_2.set_title('B. Temperatura vs Distanza -> Scost. Max', fontsize=9, fontweight='bold')

# Griglia 3: Distanza vs Flow_Rate -> Scostamento massimo (Temperatura fissa)
D3, F3, Z_scost_3, _ = calcola_griglia_3d(D_range, F_range, temp_init, 'temp')
surf3 = [ax_3d_3.plot_surface(D3, F3, Z_scost_3, cmap='inferno', alpha=0.75, edgecolor='none')]
pt3 = ax_3d_3.scatter([dist_init], [flow_init], [scost_max_init], color='red', s=60, zorder=5)
ax_3d_3.set_xlabel('D (mm)', fontsize=7)
ax_3d_3.set_ylabel('FR (%)', fontsize=7)
ax_3d_3.set_zlabel('S. Max (mm)', fontsize=7)
ax_3d_3.set_title('C. Distanza vs Flow Rate -> Scost. Max', fontsize=9, fontweight='bold')

# Griglia 4: Distanza vs Flow_Rate -> Massa (Temperatura fissa)
D4, F4, _, Z_massa_4 = calcola_griglia_3d(D_range, F_range, temp_init, 'temp')
surf4 = [ax_3d_4.plot_surface(D4, F4, Z_massa_4, cmap='magma', alpha=0.75, edgecolor='none')]
pt4 = ax_3d_4.scatter([dist_init], [flow_init], [massa_init], color='red', s=60, zorder=5)
ax_3d_4.set_xlabel('D (mm)', fontsize=7)
ax_3d_4.set_ylabel('FR (%)', fontsize=7)
ax_3d_4.set_zlabel('Massa (g)', fontsize=7)
ax_3d_4.set_title('D. Distanza vs Flow Rate -> Massa', fontsize=9, fontweight='bold')

# --------------------------------
# 6. CONTROLLI SLIDER INTERATTIVI
# --------------------------------
ax_color = 'lightgoldenrodyellow'
ax_temp = plt.axes([0.15, 0.09, 0.70, 0.02], facecolor=ax_color)
ax_dist = plt.axes([0.15, 0.05, 0.70, 0.02], facecolor=ax_color)
ax_flow = plt.axes([0.15, 0.01, 0.70, 0.02], facecolor=ax_color)

slider_temp = Slider(ax_temp, 'Temperatura (°C)', 200.0, 220.0, valinit=temp_init, valstep=0.5)
slider_dist = Slider(ax_dist, 'Distanza Linea (mm)', 0.50, 0.70, valinit=dist_init, valstep=0.01)
slider_flow = Slider(ax_flow, 'Flow Rate (%)', 80.0, 120.0, valinit=flow_init, valstep=1.0)

# --------------------------
# 7. AGGIORNAMENTO DINAMICO
# --------------------------
def update(val):
    t = slider_temp.val
    d = slider_dist.val
    f = slider_flow.val
    
    # 1. Ricalcolo Predizione
    nuovi_valori = calcola_predizione(t, d, f)
    nuova_massa = nuovi_valori[massa_col_idx]
    nuovo_scost_max = nuovi_valori[scost_max_idx]
    
    # 2. Aggiornamento Istogramma
    for bar, txt, v_nuovo in zip(bars, text_labels, nuovi_valori):
        bar.set_height(v_nuovo)
        txt.set_text(f'{v_nuovo:.3f}')
        txt.set_y(v_nuovo + 0.02)
    ax_bars.set_ylim(0, max(nuovi_valori) * 1.25)
    
    # 3. Aggiornamento Box 
    c_sem, _ = eval_semaforo(nuova_massa)
    status_box.set_text(f"RIFERIMENTO FFF: 5.0 g\n\nMassa stimata: {nuova_massa:.3f} g")
    status_box.set_bbox(dict(boxstyle='round,pad=0.8', facecolor=c_sem, alpha=0.9))
    
    # 4. Aggiornamento Superfici 3D e punti 
    global surf1, surf2, surf3, surf4, pt1, pt2, pt3, pt4
    
    # Aggiornamento Superficie 1 & Punto 1 (Distanza fissa)
    surf1[0].remove()
    pt1.remove()
    _, _, Z1, _ = calcola_griglia_3d(T_range, F_range, d, 'dist')
    surf1[0] = ax_3d_1.plot_surface(T1, F1, Z1, cmap='viridis', alpha=0.75, edgecolor='none')
    pt1 = ax_3d_1.scatter([t], [f], [nuovo_scost_max], color='red', s=60, zorder=5)
    
    # Aggiornamento Superficie 2 & Punto 2 (Flow_Rate fisso)
    surf2[0].remove()
    pt2.remove()
    _, _, Z2, _ = calcola_griglia_3d(T_range, D_range, f, 'flow')
    surf2[0] = ax_3d_2.plot_surface(T2, D2, Z2, cmap='plasma', alpha=0.75, edgecolor='none')
    pt2 = ax_3d_2.scatter([t], [d], [nuovo_scost_max], color='red', s=60, zorder=5)
    
    # Aggiornamento Superfici 3 e 4 & Punti 3 e 4 (Temperatura fissa)
    surf3[0].remove()
    surf4[0].remove()
    pt3.remove()
    pt4.remove()
    _, _, Z3, Z4 = calcola_griglia_3d(D_range, F_range, t, 'temp')
    surf3[0] = ax_3d_3.plot_surface(D3, F3, Z3, cmap='inferno', alpha=0.75, edgecolor='none')
    surf4[0] = ax_3d_4.plot_surface(D4, F4, Z4, cmap='magma', alpha=0.75, edgecolor='none')
    pt3 = ax_3d_3.scatter([d], [f], [nuovo_scost_max], color='red', s=60, zorder=5)
    pt4 = ax_3d_4.scatter([d], [f], [nuova_massa], color='red', s=60, zorder=5)
    
    fig.canvas.draw_idle()

slider_temp.on_changed(update)
slider_dist.on_changed(update)
slider_flow.on_changed(update)

plt.show()