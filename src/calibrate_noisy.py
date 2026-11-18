import json
import numpy as np
from pathlib import Path

logits_dir = Path("outputs/logits")
res_dir = Path("outputs/results")
sigmas = ["0.01", "0.0125", "0.02"]

for seed in [0, 1, 2]:
    # Charger la température
    with open(res_dir / f"metrics_val_seed{seed}.json", "r") as f:
        T = json.load(f)["temperature"]
    
    for sigma in sigmas:
        src = logits_dir / f"test_id_seed{seed}_sigma{sigma}.npy"
        if not src.exists():
            print(f"Pas trouvé : {src.name}")
            continue
            
        # Appliquer la température et sauvegarder
        logits = np.load(src)
        logits_calib = logits / T
        dst = logits_dir / f"test_id_seed{seed}_sigma{sigma}_calibrated.npy"
        np.save(dst, logits_calib.astype(np.float32))
        print(f"Créé : {dst.name} (T={T:.4f})")

print("Terminé !")