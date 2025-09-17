# Industrial Defect Detection

Industrial surface defect detection using a Convolutional Autoencoder on the MVTec AD dataset.

## Features
- Deep learning with PyTorch (Conv Autoencoder)
- Training and inference scripts
- Heatmap visualization for defects
- Supports MVTec AD dataset (e.g. metal_nut, capsule)

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Training
```bash
python src/train.py --dataset /path/to/mvtec/metal_nut --epochs 30 --save models/conv_ae.pt
```

## Inference
```bash
python src/infer.py --image /path/to/mvtec/metal_nut/test/defective/xxx.png --model models/conv_ae.pt --save outputs/heatmap.png
```

## License
MIT
