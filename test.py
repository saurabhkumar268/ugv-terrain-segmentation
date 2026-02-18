import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# --- CONFIGURATION ---
# Path to your trained model
MODEL_PATH = os.path.join('runs', 'segmentation_model.pth')
# Path to the test images (from Hackathon dataset)
TEST_DIR = os.path.join('dataset', 'testimages')
# Where to save the results
OUTPUT_DIR = 'predictions'
# Same size as training
IMG_SIZE = (256, 256)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- VISUALIZATION COLORS ---
# Mapping class indices (0-9) to RGB colors for easy viewing
# 0:Trees, 1:Lush Bushes, 2:Dry Grass, 3:Dry Bushes, 4:Ground Clutter
# 5:Flowers, 6:Logs, 7:Rocks, 8:Landscape, 9:Sky
PALETTE = np.array([
    [34, 139, 34],   # 0: Forest Green (Trees)
    [0, 255, 0],     # 1: Lime Green (Lush Bushes)
    [218, 165, 32],  # 2: Goldenrod (Dry Grass)
    [139, 69, 19],   # 3: Saddle Brown (Dry Bushes)
    [128, 128, 128], # 4: Gray (Clutter)
    [255, 105, 180], # 5: Hot Pink (Flowers)
    [101, 67, 33],   # 6: Dark Brown (Logs)
    [64, 64, 64],    # 7: Dark Gray (Rocks)
    [244, 164, 96],  # 8: Sandy Brown (Landscape)
    [135, 206, 235]  # 9: Sky Blue (Sky)
], dtype=np.uint8)

# --- U-NET MODEL DEFINITION ---
# (Must match train.py exactly to load weights correctly)
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

# --- INFERENCE FUNCTION ---
def run_test():
    # 1. Check for Model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        print("   Did you run train.py successfully?")
        return

    # 2. Check for Images
    if not os.path.exists(TEST_DIR):
        print(f"❌ Error: Test images not found at {TEST_DIR}")
        print("   Check your folder structure.")
        return

    # 3. Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Loading model from {MODEL_PATH}...")
    
    # Load Model
    model = UNet(n_classes=10).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except RuntimeError as e:
        print(f"❌ Error loading weights: {e}")
        return
        
    model.eval() # Set to evaluation mode
    
    # Image Transforms (Must match training normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_images = sorted(os.listdir(TEST_DIR))
    print(f"📂 Found {len(test_images)} images. Processing...")

    # 4. Processing Loop
    with torch.no_grad():
        for i, img_name in enumerate(test_images):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(TEST_DIR, img_name)
            
            # Open and Resize
            original_image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = original_image.size
            
            input_img = original_image.resize(IMG_SIZE, Image.BILINEAR)
            input_tensor = transform(input_img).unsqueeze(0).to(device)
            
            # Predict
            output = model(input_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
            # Colorize
            # Create an empty RGB image
            color_mask = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
            
            # Apply colors from PALETTE
            for cls_idx in range(10):
                color_mask[pred_mask == cls_idx] = PALETTE[cls_idx]
            
            # Resize back to original size for high-quality output
            result_image = Image.fromarray(color_mask)
            result_image = result_image.resize((orig_w, orig_h), Image.NEAREST)
            
            # Save
            save_path = os.path.join(OUTPUT_DIR, f"pred_{img_name}")
            result_image.save(save_path)
            
            if i % 5 == 0:
                print(f"   Saved {save_path}")

    print(f"🎉 Done! Check the '{OUTPUT_DIR}' folder for your results.")

if __name__ == "__main__":
    run_test()