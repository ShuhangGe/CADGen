# CADGen: 3D CAD Model Generation Framework

CADGen is a deep learning framework for generating 3D CAD models from multi-view images. The project focuses on converting 2D representations (front, top, and side views) into 3D CAD models with precise command sequences and parameters.

## Features

- Multi-view image to 3D CAD model conversion
- Various model architectures:
  - Autoregressive models
  - Set Transformer-based models
  - Deformable models
- Point cloud generation and processing
- Comprehensive evaluation metrics

## Project Structure

```
CADGen/
├── 3dmodel/                     # Core 3D model generation code
│   ├── model/                   # Model architectures
│   ├── datasets/                # Dataset handling
│   ├── cadlib/                  # CAD processing utilities
│   ├── utils/                   # Helper functions
│   ├── train.py                 # Main training script
│   ├── train_deformable.py      # Training for deformable models
│   ├── train_deformable_cad.py  # Training for deformable CAD models
│   └── test.py                  # Testing script
├── 3dmodel_autoregressive/      # Autoregressive model implementations
├── 3dmodel_autoregressive_pointcloud/ # Point cloud autoregressive models
├── 3dmodel_settransformer/      # Set Transformer model implementations
├── data_process/                # Data preprocessing utilities
│   ├── json2pc_my.py            # JSON to point cloud conversion
│   ├── json2vec_my.py           # JSON to vector conversion
│   └── various data conversion scripts
├── evaluation/                  # Evaluation metrics and scripts
│   ├── evaluate_ae_cd.py        # Chamfer distance evaluation
│   ├── evaluate_ae_acc.py       # Accuracy evaluation
│   └── evaluate_gen_torch.py    # Generation evaluation
└── bulletpoints/                # Additional utilities
```

## Requirements

- PyTorch
- CUDA-enabled GPU
- NumPy
- timm (PyTorch Image Models)
- Tensorboard

## Installation

Clone the repository:
```bash
git clone https://github.com/your-username/CADGen.git
cd CADGen
```

Install the required packages (consider creating a virtual environment first):
```bash
pip install torch torchvision numpy tensorboard timm
```

## Usage

### Data Preparation

The data processing scripts in the `data_process/` directory help prepare your data:

```bash
python data_process/data_split.py
python data_process/json2pc_my.py
```

### Training

To train the basic model:
```bash
python 3dmodel/train.py
```

For deformable model training:
```bash
python 3dmodel/train_deformable.py
```

For deformable CAD model training:
```bash
python 3dmodel/train_deformable_cad.py
```

### Evaluation

Evaluate your trained models using the scripts in the `evaluation/` directory:

```bash
python evaluation/evaluate_gen_torch.py
python evaluation/evaluate_ae_cd.py
```

## Model Architecture

The CADGen framework uses a multi-stage architecture:
1. Image feature extraction using ResNet backbones
2. Feature fusion from multiple views
3. Transformer-based processing
4. Command and parameter prediction for CAD model generation

## License

[Add your license information here]

## Acknowledgements

[Add acknowledgements, citations, or references to papers that inspired this work]

## Contact

[Add contact information]
