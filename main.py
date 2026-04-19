# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# ----------------------------
# 1. Set up the device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# 2. Data Preprocessing & Loader
# ----------------------------
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

data_dir = Path("dogs_vs_cats_small")
image_datasets = {
    x: datasets.ImageFolder(root=data_dir / x, transform=data_transforms[x])
    for x in ['train', 'val']
}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=32, shuffle=(x == 'train'), num_workers=2)
    for x in ['train', 'val']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes  # ['cat', 'dog']

print(f"Train size: {dataset_sizes['train']}, Val size: {dataset_sizes['val']}")
print(f"Classes: {class_names}")

# ----------------------------
# 3. Define the training function
# ----------------------------
def train_model(model, criterion, optimizer, num_epochs=5):
    model.to(device)
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Training
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']
        train_acc_history.append(epoch_acc.item())

        # Validation
        model.eval()
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)

        val_acc = val_running_corrects.double() / dataset_sizes['val']
        val_acc_history.append(val_acc.item())

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Acc: {val_acc:.4f}')

    return model, train_acc_history, val_acc_history

# ----------------------------
# 4. Experiment 1: Fine-tuning
# ----------------------------
print("\nStarting the fine-tuning experiment...")
model_ft = models.resnet18(pretrained=True) 
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

model_ft, train_acc_ft, val_acc_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs=5)

# ----------------------------
# 5. Experiment 2: Training from Scratch
# ----------------------------
print("\nStart training the experiment from scratch...")
model_scratch = models.resnet18(pretrained=False)
num_ftrs = model_scratch.fc.in_features
model_scratch.fc = nn.Linear(num_ftrs, 2)

optimizer_scratch = optim.Adam(model_scratch.parameters(), lr=0.001)

model_scratch, train_acc_s, val_acc_s = train_model(model_scratch, criterion, optimizer_scratch, num_epochs=5)

# ----------------------------
# 6. Plot the accuracy comparison chart (Figure 1)
# ----------------------------
epochs = list(range(1, 6))
plt.figure(figsize=(8, 5))
plt.plot(epochs, val_acc_ft, 'o-', label='Fine-tuning (Val)', color='blue')
plt.plot(epochs, val_acc_s, 's--', label='Scratch (Val)', color='red')
plt.title('Accuracy: Fine-tuning vs. Scratch')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("accuracy_comparison.png", dpi=150, bbox_inches='tight')

# ----------------------------
# 7. Visualization of prediction results (Figure 2)
# ----------------------------
def visualize_predictions(model, num_images=6):
    was_training = model.training
    model.eval()

    fig = plt.figure(figsize=(12, 8))
    images_so_far = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                if images_so_far >= num_images:
                    break
                ax = plt.subplot(num_images//2, 2, images_so_far + 1)
                ax.axis('off')
                
                pred_label = class_names[preds[j]]
                true_label = class_names[labels[j]]
                ax.set_title(f'pred: {pred_label}\ntrue: {true_label}', fontsize=12)
                
                img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                plt.imshow(img)
                images_so_far += 1
            if images_so_far >= num_images:
                break

    model.train(mode=was_training)
    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=150, bbox_inches='tight')

visualize_predictions(model_ft)

# ----------------------------
# 8. Print the final result summary
# ----------------------------
print("\nExperiment completed:")
print(f"Final validation accuracy of the fine-tuned model: {val_acc_ft[-1]:.4f}")
print(f"Final validation accuracy of the model trained from scratch:{val_acc_s[-1]:.4f}")
print(f"Fine-tuning is superior to training from scratch: {(val_acc_ft[-1] - val_acc_s[-1])*100:.2f}%")

torch.save(model_ft.state_dict(), 'fine_tuned_model.pth')
torch.save(model_scratch.state_dict(), 'scratch_model.pth')