import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

DATA_DIR = os.getenv("DATA_DIR", "dataset/chest_xray")
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("artifacts", exist_ok=True)

train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class_names = train_ds.classes
with open("artifacts/class_names.json", "w") as f:
    json.dump(class_names, f)

model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
model.classifier = nn.Linear(model.classifier.in_features, len(class_names))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_acc = 0.0
best_w = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):
    model.train()
    tr_correct = tr_total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tr_correct += (out.argmax(1) == y).sum().item()
        tr_total += y.size(0)

    model.eval()
    va_correct = va_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            va_correct += (out.argmax(1) == y).sum().item()
            va_total += y.size(0)

    tr_acc = tr_correct / tr_total
    va_acc = va_correct / va_total
    print(f"Epoch {epoch+1}/{EPOCHS} train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")

    if va_acc > best_acc:
        best_acc = va_acc
        best_w = copy.deepcopy(model.state_dict())

model.load_state_dict(best_w)
torch.save(model.state_dict(), "artifacts/best_model.pth")
print("Saved model to artifacts/best_model.pth")
print("Best val accuracy:", best_acc)
