import os

# 1. Where are we right now?
current_dir = os.getcwd()
print(f"📍 You are currently in: {current_dir}")

# 2. What files are next to the script?
print(f"📂 Files here: {os.listdir(current_dir)}")

# 3. specific check for the dataset
expected_path = os.path.join(current_dir, 'dataset')
if os.path.exists(expected_path):
    print(f"✅ 'dataset' folder found.")
    print(f"   Contents: {os.listdir(expected_path)}")
    
    # Check deeper for Train
    train_path = os.path.join(expected_path, 'Train')
    if os.path.exists(train_path):
        print(f"✅ 'Train' folder found.")
    else:
        print(f"❌ ERROR: 'dataset' exists, but 'Train' is missing. You might have a double folder (dataset/dataset).")
else:
    print(f"❌ ERROR: Python cannot find a folder named 'dataset' here.")
    print("   Did you name it something else? (e.g., 'hackathon_data' or 'data')")