import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.autoencoder import ConvAE
from src.data import MVTecDataset
import argparse, os

def train(dataset_path, epochs, save_path):
    dataset = MVTecDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = ConvAE()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0
        for imgs, _ in dataloader:
            recon = model(imgs)
            loss = criterion(recon, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to MVTec dataset class folder")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--save", type=str, default="models/conv_ae.pt")
    args = parser.parse_args()
    train(args.dataset, args.epochs, args.save)
