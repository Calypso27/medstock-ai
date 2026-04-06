import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
from huggingface_hub import HfApi

from utils.config import CNN_CLASSES, HF_MODEL_REPO, MODEL_FILENAME


TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def train(data_dir, epochs=20, batch_size=32, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    full_dataset = datasets.ImageFolder(data_dir, transform=TRAIN_TRANSFORM)
    assert sorted(full_dataset.classes) == sorted(CNN_CLASSES), (
        f"Classes trouvées : {full_dataset.classes} — attendu : {CNN_CLASSES}"
    )
    print(f"Classes : {full_dataset.classes} | Images : {len(full_dataset)}")


    val_size   = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = VAL_TRANSFORM

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=len(CNN_CLASSES))
    model = model.to(device)

    # on gèle le backbone et on entraîne seulement le classifier
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * imgs.size(0)
            train_correct += (out.argmax(1) == labels).sum().item()

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out  = model(imgs)
                loss = criterion(out, labels)
                val_loss    += loss.item() * imgs.size(0)
                val_correct += (out.argmax(1) == labels).sum().item()

        train_acc = train_correct / train_size * 100
        val_acc   = val_correct   / val_size   * 100
        scheduler.step()

        print(f"Epoch {epoch:02d}/{epochs} | train acc: {train_acc:.1f}% | val acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/{MODEL_FILENAME}")
            print(f"  >> modele sauvegarde ({val_acc:.1f}%)")

    print(f"Termine. Meilleure val acc : {best_val_acc:.1f}%")
    return model


def upload_to_hub():
    api = HfApi()
    model_path = f"models/{MODEL_FILENAME}"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} introuvable — lancez l'entraînement d'abord.")
    print(f"Upload vers {HF_MODEL_REPO}…")
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=MODEL_FILENAME,
        repo_id=HF_MODEL_REPO,
        repo_type="model",
    )
    print("Upload terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs",   type=int,   default=20)
    parser.add_argument("--batch",    type=int,   default=32)
    parser.add_argument("--lr",       type=float, default=1e-4)
    parser.add_argument("--upload",   action="store_true")
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch, args.lr)

    if args.upload:
        upload_to_hub()
