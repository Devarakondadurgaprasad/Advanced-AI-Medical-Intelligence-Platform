import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

DATA_DIR = os.getenv("DATA_DIR", "dataset/chest_xray")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=tf)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

with open("artifacts/class_names.json", "r") as f:
    class_names = json.load(f)

model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, len(class_names))
model.load_state_dict(torch.load("artifacts/best_model.pth", map_location=device))
model.to(device)
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        out = model(x)
        p = out.argmax(1).cpu().numpy().tolist()
        y_pred.extend(p)
        y_true.extend(y.numpy().tolist())

acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1:", f1)
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
