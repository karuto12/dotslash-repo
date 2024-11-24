import os
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

# Define a custom dataset for training
class RetailDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_labels = self.load_labels(labels_file)

    def load_labels(self, labels_file):
        labels = {}
        with open(labels_file, 'r') as f:
            for line in f:
                image, label = line.strip().split(',')
                labels[image] = int(label)
        return labels

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image_name = list(self.image_labels.keys())[idx]
        label = self.image_labels[image_name]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Fine-tune a pre-trained model
class EmbeddingModel:
    def __init__(self, num_classes, device):
        self.device = device
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Replace the classifier
        self.model = self.model.to(device)

    def train(self, train_loader, epochs, lr, save_path):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def generate_embeddings(self, image_dir, transform, output_path):
        self.model.eval()
        embeddings = {}

        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            image = Image.open(image_path).convert("RGB")
            if transform:
                image = transform(image)
            image = image.unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model(image).cpu().numpy()
            embeddings[image_name] = embedding

        np.save(output_path, embeddings)
        print(f"Embeddings saved to {output_path}")

# Main function for training or embedding generation
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.mode == "train":
        dataset = RetailDataset(args.image_dir, args.labels_file, transform=transform)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        model = EmbeddingModel(num_classes=args.num_classes, device=device)
        model.train(train_loader, epochs=args.epochs, lr=args.lr, save_path=args.model_path)
    elif args.mode == "embed":
        model = EmbeddingModel(num_classes=args.num_classes, device=device)
        model.model.load_state_dict(torch.load(args.model_path))
        model.generate_embeddings(args.image_dir, transform, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model or generate embeddings.")
    parser.add_argument("--mode", choices=["train", "embed"], required=True, help="Mode: 'train' or 'embed'.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory with images.")
    parser.add_argument("--labels_file", type=str, help="Path to the labels file for training.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save or load the model.")
    parser.add_argument("--output_path", type=str, help="Path to save embeddings.")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of product categories.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    args = parser.parse_args()
    main(args)
