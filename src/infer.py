import torch
import cv2
import numpy as np
from src.autoencoder import ConvAE
from torchvision import transforms
from PIL import Image
import argparse, os

def infer(image_path, model_path, save_path):
    model = ConvAE()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        recon = model(x)
    diff = torch.abs(recon - x).squeeze().permute(1, 2, 0).numpy()
    heatmap = (diff / diff.max() * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, heatmap)
    print(f"Saved heatmap to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--model", required=True, help="Path to trained model weights")
    parser.add_argument("--save", default="outputs/heatmap.png")
    args = parser.parse_args()
    infer(args.image, args.model, args.save)
