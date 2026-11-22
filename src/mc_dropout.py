# src/mc_dropout.py
import torch
import numpy as np
import os

def enable_dropout(model):
    """
    Active le Dropout uniquement pour les couches Dropout pendant l'inférence.
    Les autres couches (comme BatchNorm) restent en mode évaluation.
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def predict_with_mc_dropout(model, dataloader, device, num_passes=30):
    """
    Effectue N passes d'inférence avec Dropout activé pour estimer l'incertitude.
    Retourne les probabilités moyennes, les vraies étiquettes et les IDs.
    """
    model.eval() # Mode évaluation par défaut
    enable_dropout(model) # Force le Dropout à rester actif
    
    all_probs = []
    all_labels = []
    all_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Adapte les clés 'image', 'label', 'id' selon votre dataset.py
            images = batch['image'].to(device)
            labels = batch['label'].cpu().numpy()
            ids = batch['id'] if isinstance(batch['id'], list) else batch['id'].cpu().numpy()
            
            batch_probs = []
            for _ in range(num_passes):
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                batch_probs.append(probs.cpu().numpy())
            
            # Moyenne des probabilités sur les N passes (prédiction finale)
            mean_probs = np.mean(batch_probs, axis=0)
            
            all_probs.append(mean_probs)
            all_labels.append(labels)
            all_ids.extend(ids)
            
    return np.concatenate(all_probs), np.concatenate(all_labels), np.array(all_ids)

def save_predictions_npz(probs, labels, ids, shift_type, seed, output_dir="outputs"):
    """
    Sauvegarde les résultats au format .npz standardisé pour l'équipe.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = f"{output_dir}/mc_dropout_{shift_type}_seed{seed}.npz"
    
    np.savez(
        filepath, 
        y_true=labels, 
        y_prob=probs, 
        sample_id=ids, 
        shift_type=shift_type, 
        seed=seed
    )
    print(f" Sauvegardé avec succès : {filepath}")