import os

# Define the path exactly where test.py looks
test_dir = os.path.join('dataset', 'testimages')

print(f"🕵️ Checking path: {os.path.abspath(test_dir)}")

if os.path.exists(test_dir):
    files = os.listdir(test_dir)
    print(f"✅ Folder found! It contains {len(files)} items.")
    
    if len(files) > 0:
        print("   Here are the first 5 items:")
        for f in files[:5]:
            print(f"   - {f}")
            
        # Check if they are folders or images
        first_item = os.path.join(test_dir, files[0])
        if os.path.isdir(first_item):
            print("\n⚠️ WARNING: The first item is a FOLDER, not an image!")
            print("   You need to move the images out of that subfolder.")
    else:
        print("❌ ERROR: The folder is empty. Did you copy the test images there?")
else:
    print("❌ ERROR: The 'dataset/testimages' folder does not exist.")