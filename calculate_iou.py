import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from train import UNet, DesertDataset # Imports from your train.py
from torchvision import transforms

# --- CONFIGURATION ---
DATA_DIR = 'dataset'       # Make sure this matches your folder name
MODEL_PATH = 'runs/segmentation_model.pth'
BATCH_SIZE = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_iou(pred, target, n_classes=10):
    ious = []
    # Flatten predictions and targets
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    
    # Calculate IoU for each class (0-9)
    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        
        if union > 0:
            ious.append(intersection / union)
            
    # Return the average IoU of all classes present
    return np.mean(ious)

def evaluate():
    # 1. Load Data (Validation Set ONLY)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # We use split='Val' because it has masks!
        val_dataset = DesertDataset(DATA_DIR, split='Val', transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print(f"✅ Loaded {len(val_dataset)} validation images for evaluation.")
    except Exception as e:
        print(f"❌ Error loading validation data: {e}")
        return

    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found. Train it first!")
        return
        
    model = UNet(n_classes=10).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    # 3. Calculate IoU
    total_iou = 0
    num_batches = 0
    
    print("⏳ Calculating IoU... (This may take a minute)")
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            outputs = model(images)
            batch_iou = calculate_iou(outputs, masks)
            
            total_iou += batch_iou
            num_batches += 1
            
    # 4. Final Result
    final_iou = total_iou / num_batches
    print(f"\n🏆 FINAL IoU SCORE: {final_iou:.4f}")
    print(f"📝 Put this number in your Hackathon Report under 'Results'.")

if __name__ == '__main__':
    evaluate()