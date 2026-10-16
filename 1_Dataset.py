import pandas as pd

# ------------------------------------------------------------------------
# 1. CARICAMENTO DEI DATI NUMERICI RELATIVI A 81 STAMPE MEDIANTE IL ROBOT
# ------------------------------------------------------------------------
NOME_FILE_EXCEL = 'Dataset_cubetti.xlsx'

# ------------------------------
# 2. CARICAMENTO DEL FILE EXCEL
# ------------------------------
try:
    df = pd.read_excel(NOME_FILE_EXCEL)
    print(f" File '{NOME_FILE_EXCEL}' caricato con successo!\n")
except Exception as e:
    print(f" Errore nel caricamento del file Excel: {e}")
    exit()

# -------------------------------------------------------
# 3. CONTROLLO DELLE DIMENSIONI (DEVONO ESSERE 81 RIGHE)
# -------------------------------------------------------
print(f"Totale righe (campioni): {len(df)}")
print("Nomi delle colonne trovate:")
print(list(df.columns))

# ---------------------------------
# 4. ANTEPRIMA DELLE PRIME 5 RIGHE
# ---------------------------------
print("\n--- Anteprima dati ---")
print(df.head(5))

# ------------------------------------------
# 5. SALVATAGGIO IN FORMATO CSV PER PyTorch
# ------------------------------------------
df.to_csv('Dataset_cubetti.csv', index=False)
print("\n Dataset convertito e salvato come 'Dataset_cubetti.csv'")