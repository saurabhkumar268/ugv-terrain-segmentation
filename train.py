import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import time

# --- CONFIGURATION (RTX 3050 OPTIMIZED) ---
DATA_DIR = 'dataset'       
BATCH_SIZE = 4             # RTX 3050 usually has 4GB/6GB VRAM. Keep this at 4 to avoid "Out of Memory" crashes.
LEARNING_RATE = 1e-4
EPOCHS = 25                # As requested
IMG_SIZE = (256, 256)

# --- GPU SETUP ---
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f"✅ SUCCESS: Detected NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    print("   🚀 Training will be fast!")
else:
    DEVICE = torch.device('cpu')
    print("❌ WARNING: CUDA not detected. Running on CPU (Slow).")
    print("   Make sure you installed the CUDA version of PyTorch.")

# --- 1. DATASET CLASS ---
class DesertDataset(Dataset):
    def __init__(self, root_dir, split='Train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.masks_dir = os.path.join(root_dir, split, 'masks')
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"❌ Error: Folder not found at {self.images_dir}")
            
        print(f"   📂 Scanning: {self.images_dir}...")
        self.images = [
            f for f in sorted(os.listdir(self.images_dir)) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if len(self.images) == 0:
            raise RuntimeError(f"❌ Error: No images found!")
            
        print(f"   ✅ Found {len(self.images)} valid images. Loading to GPU...")
        
        self.id_map = {
            100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 
            600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
        }

    def __len__(self):
        return len(self.images)

    def map_mask(self, mask):
        mask_np = np.array(mask)
        new_mask = np.zeros_like(mask_np, dtype=np.int64)
        for k, v in self.id_map.items():
            new_mask[mask_np == k] = v
        return new_mask

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path)
        except Exception as e:
            print(f"⚠️ Warning: Error opening {img_name}. Skipping.")
            return torch.zeros((3, 256, 256)), torch.zeros((256, 256)).long()
        
        image = image.resize(IMG_SIZE, Image.BILINEAR)
        mask = mask.resize(IMG_SIZE, Image.NEAREST)

        mask_np = self.map_mask(mask)

        if self.transform:
            image = self.transform(image)
        
        mask_tensor = torch.from_numpy(mask_np).long()
        return image, mask_tensor

# --- 2. U-NET MODEL ---
class UNet(nn.Module):
    def __init__(self, n_classes=10):
        super(UNet, self).__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        
        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = conv_block(512 + 256, 256)
        self.dec2 = conv_block(256 + 128, 128)
        self.dec1 = conv_block(128 + 64, 64)
        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.final(d1)

# --- 3. TRAINING LOOP ---
def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("⏳ Initializing dataset...")
    try:
        train_dataset = DesertDataset(DATA_DIR, split='Train', transform=transform)
        # pin_memory=True speeds up transfer to your RTX 3050
        # num_workers=0 avoids Windows multiprocessing bugs
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    except Exception as e:
        print(f"❌ Critical Data Error: {e}")
        return

    model = UNet(n_classes=10).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print(f"🚀 Starting training for {EPOCHS} epochs on {DEVICE}...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for i, (images, masks) in enumerate(train_loader):
            # Move data to RTX 3050
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i > 0 and i % 50 == 0:
                print(f"   Epoch {epoch+1}: Batch {i} done...")
            
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start_time
        print(f"✨ Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f} | Total Time: {elapsed:.0f}s")

    # Save Model
    os.makedirs('runs', exist_ok=True)
    save_path = os.path.join('runs', 'segmentation_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"🎉 Training Complete! Model saved to {save_path}")

if __name__ == '__main__':
    train()