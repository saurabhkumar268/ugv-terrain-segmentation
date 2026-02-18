import matplotlib.pyplot as plt

# REPLACE these with the actual numbers from your training terminal!
# Example: Epoch [1/25], Loss: 0.8521, Val IoU: 0.312
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
loss_values = [0.8855, 0.5986, 0.5092, 0.4591, 0.4276, 0.4015, 0.3887, 0.3732, 0.3556, 0.3644, 0.3468, 0.3312, 0.3252, 0.3374, 0.3143, 0.3102, 0.3233, 0.3052, 0.2988, 0.3035, 0.2952, 0.3015, 0.3075, 0.2856, 0.2829]
iou_values = [0.1135, 0.4014, 0.4908, 0.4509, 0.5724, 0.5985, 0.6113, 0.6268, 0.6444, 0.6356, 0.6532, 0.6688, 0.6748, 0.6626, 0.6857, 0.6898, 0.6767, 0.6948, 0.7012, 0.6965, 0.7048, 0.6985, 0.6925, 0.7144, 0.7171]

# 1. Plot Training Loss
plt.figure(figsize=(25, 5))
plt.plot(epochs, loss_values, label='Training Loss', color='red', marker='o')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_graph.png') # Saves the image for your report
plt.show()

# 2. Plot IoU Score
plt.figure(figsize=(25, 5))
plt.plot(epochs, iou_values, label='Validation IoU', color='blue', marker='o')
plt.title('Model Accuracy (IoU) over Epochs')
plt.xlabel('Epochs')
plt.ylabel('IoU Score')
plt.legend()
plt.grid(True)
plt.savefig('iou_graph.png') # Saves the image for your report
plt.show()

print("✅ Graphs saved! Check loss_graph.png and iou_graph.png")