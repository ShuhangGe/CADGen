# CADGen: Advanced 3D CAD Model Generation Framework

CADGen is a state-of-the-art deep learning framework for generating 3D CAD models from multi-view images. The project enables the conversion of 2D representations (front, top, and side views) into precise 3D CAD models with accurate command sequences and parameters.

<p align="center">
  <img src="docs/assets/cadgen-overview.png" alt="CADGen Overview" width="800"/>
  <br>
  <em>CADGen: Generating 3D CAD models from 2D multi-view inputs</em>
</p>

## 🔍 Features

- **Multi-view to 3D Conversion**: Transform 2D images into complete 3D CAD models
- **Multiple Model Architectures**:
  - 🔹 Autoregressive models for sequential generation
  - 🔹 Set Transformer-based models for global reasoning
  - 🔹 Deformable models for complex geometry understanding
- **Point Cloud Processing**: Advanced point cloud generation and manipulation
- **Comprehensive Evaluation**: Robust metrics for assessing generation quality

## 🏗️ Project Structure

The CADGen project is organized into several interconnected modules, each with a specific role in the 3D CAD generation pipeline:

### Core Modules

```
CADGen/
├── 3dmodel/                     # Core model implementation
│   ├── model/                   # Model architecture definitions
│   │   ├── model.py             # Base model definitions
│   │   ├── model_simple.py      # Simplified model variant
│   │   ├── transformer.py       # Transformer components
│   │   └── loss.py              # Loss functions
│   ├── datasets/                # Dataset handling for training
│   │   ├── dataset.py           # Core dataset classes
│   │   └── augmentation.py      # Data augmentation strategies
│   ├── cadlib/                  # CAD processing and manipulation
│   │   ├── cad_utils.py         # CAD operation utilities
│   │   └── command_parser.py    # CAD command parsing
│   ├── utils/                   # Utility functions
│   │   ├── config.py            # Configuration handling
│   │   ├── logger.py            # Logging utilities
│   │   └── vis_utils.py         # Visualization helpers
│   ├── train.py                 # Base model training script
│   ├── train_deformable.py      # Deformable model training
│   ├── train_deformable_cad.py  # Deformable CAD model training
│   └── test.py                  # Model testing and evaluation
```

### Model Variants

The project includes specialized model architectures, each designed to address specific aspects of the CAD generation task:

```
CADGen/
├── 3dmodel_autoregressive/      # Sequential CAD generation models
│   ├── model/                   # Autoregressive architecture
│   ├── train.py                 # Training script
│   └── inference.py             # Inference utilities
│
├── 3dmodel_autoregressive_pointcloud/ # Point cloud-based autoregressive models
│   ├── model/                   # Point cloud sequential model
│   └── train.py                 # Training script
│
├── 3dmodel_settransformer/      # Set Transformer implementation
│   ├── model/                   # Set-based architecture
│   └── train.py                 # Training script
```

### Data Processing Pipeline

Tools for preparing, converting, and processing CAD data:

```
CADGen/
├── data_process/                # Data preprocessing and conversion
│   ├── json2pc_my.py            # Converts JSON CAD to point clouds
│   ├── json2vec_my.py           # Converts JSON CAD to vector representations
│   ├── data_split.py            # Splits datasets for training/validation
│   ├── ply2txt.py               # PLY to text format conversion
│   └── txt2np.py                # Text to NumPy format conversion
```

### Evaluation Framework

Comprehensive evaluation tools for assessing model performance:

```
CADGen/
├── evaluation/                  # Model evaluation tools
│   ├── evaluate_ae_cd.py        # Chamfer distance evaluation
│   ├── evaluate_ae_acc.py       # Command accuracy evaluation
│   ├── evaluate_gen_torch.py    # Generation quality evaluation
│   └── utils/                   # Evaluation utilities
│       ├── metrics.py           # Metrics implementation
│       └── visualization.py     # Results visualization
```

### Advanced Approaches

Specialized modules for advanced techniques:

```
CADGen/
└── bulletpoints/                # Specialized modeling approaches
    └── mae_cad/                 # Masked Autoencoder for CAD modeling
        ├── models_command.py    # Command prediction models
        ├── models_parameter_*.py # Parameter prediction variants
        ├── main_*.py            # Training scripts
        └── README.md            # Module-specific documentation
```

### Data Flow

The overall data flow in the CADGen framework follows this pattern:

1. **Data Preparation**: Raw CAD models → Processed command sequences and point clouds
2. **Model Training**: Multi-view images + CAD data → Trained models
3. **Inference**: New multi-view images → Generated 3D CAD models
4. **Evaluation**: Generated models vs. Ground truth → Performance metrics

## ⚙️ Requirements

- PyTorch 1.9+
- CUDA-enabled GPU (8GB+ recommended)
- NumPy
- timm (PyTorch Image Models)
- Tensorboard

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/CADGen.git
   cd CADGen
   ```

2. **Install dependencies**:
   ```bash
   # Create a virtual environment (recommended)
   python -m venv cadgen-env
   source cadgen-env/bin/activate  # On Windows: cadgen-env\Scripts\activate
   
   # Install required packages
   pip install torch==1.9.0 torchvision==0.10.0 numpy==1.20.3 tensorboard==2.6.0 timm==0.4.12
   ```

## 📊 Usage

### Data Preparation

Prepare your data using the processing scripts:

```bash
# Split your dataset
python data_process/data_split.py --input /path/to/data --output /path/to/output

# Convert JSON CAD data to point clouds
python data_process/json2pc_my.py --input /path/to/json --output /path/to/pointclouds
```

### Training

Train different model variants:

```bash
# Basic model training
python 3dmodel/train.py --config configs/base_config.yaml

# Deformable model training
python 3dmodel/train_deformable.py --config configs/deformable_config.yaml

# Deformable CAD model training
python 3dmodel/train_deformable_cad.py --config configs/deformable_cad_config.yaml
```

### Evaluation

Evaluate your trained models:

```bash
# Generation quality evaluation
python evaluation/evaluate_gen_torch.py --model_path /path/to/model --test_data /path/to/test_data

# Chamfer distance evaluation
python evaluation/evaluate_ae_cd.py --model_path /path/to/model --test_data /path/to/test_data
```

## 🧠 Model Architectures

The CADGen framework implements several powerful architectures:

### Base Model (Views2Points)

The base model takes multi-view images and generates point clouds representing the 3D shape:

- **Backbone**: ResNet-based feature extraction from each view
- **Fusion Module**: Multi-view feature integration
- **Decoder**: Transformer architecture that outputs 3D points and CAD command sequences

<p align="center">
  <img src="docs/assets/base-model.png" alt="Base Model Architecture" width="600"/>
  <br>
  <em>Views2Points model architecture</em>
</p>

### Deformable Models

The deformable variants incorporate deformable attention mechanisms for better geometric understanding:

- **Deformable Attention**: 3D-aware attention mechanisms that enhance spatial reasoning
- **Point Sampling Strategy**: Adaptive point sampling to focus computation on relevant regions
- **Hierarchical Processing**: Multi-scale feature processing for capturing both global and local details

### Autoregressive Approach

The autoregressive model generates CAD commands sequentially:

- **Command Sequence Modeling**: LSTM/Transformer-based sequence generation
- **Parameter Prediction**: Specialized networks for predicting geometric parameters
- **Joint Learning**: Combined learning of commands and their parameters

## 📋 Dataset Format

### Input Images

- Front, top, and side view images (RGB or grayscale)
- Resolution: Recommend 256×256 pixels
- Format: PNG or JPG

### 3D CAD Data

- Point cloud representations (1024 points recommended)
- CAD command sequences in JSON format:
  ```json
  {
    "commands": ["line", "circle", "extrude", ...],
    "parameters": [[x1, y1, z1, x2, y2, z2], [cx, cy, r], [h], ...]
  }
  ```

### Data Organization

```
data/
├── images/
│   ├── front/
│   ├── top/
│   └── side/
├── point_clouds/
└── cad_commands/
```

## 📈 Performance Metrics

The framework evaluates performance using:

1. **Chamfer Distance (CD)**: Measures the similarity between generated and ground truth point clouds
2. **Command Accuracy**: Percentage of correctly predicted CAD commands
3. **Parameter Error**: Mean squared error between predicted and ground truth parameters
4. **Visual Quality**: Qualitative assessment of the generated 3D models

### Benchmark Results

| Model | CD ↓ | Cmd Acc ↑ | Param MSE ↓ |
|-------|------|-----------|-------------|
| Base  | -    | -         | -           |
| Deformable | - | -       | -           |
| Autoregressive | - | -   | -           |

## 🖥️ Inference Examples

Generate a CAD model from input views:

```python
import torch
from model.model_simple import Views2Points

# Load model
model = Views2Points(args)
model.load_state_dict(torch.load('path/to/model_checkpoint'))
model.eval()

# Load input images
front_img = load_image('front.png')
top_img = load_image('top.png')
side_img = load_image('side.png')

# Generate CAD model
with torch.no_grad():
    output = model(front_img, top_img, side_img)
    
# Extract commands and parameters
commands = output['pred_commands']
parameters = output['pred_args']

# Save or visualize the result
save_cad_model(commands, parameters, 'output.obj')
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in the config
   - Use model variants with fewer parameters
   - Enable gradient checkpointing

2. **Slow Training**
   - Use mixed precision training (enabled by default)
   - Optimize data loading with more workers
   - Check for bottlenecks using profiling tools

3. **Poor Generation Quality**
   - Ensure training data quality and diversity
   - Try different model variants
   - Adjust loss function weights in `CADLoss` class

### Environment Setup Issues

For environment setup problems, make sure you have:
- CUDA toolkit compatible with your PyTorch version
- Sufficient GPU memory (8GB+ recommended)
- All dependencies installed with compatible versions

## 🤝 Contributing

Contributions to CADGen are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the project's style guidelines and includes appropriate tests.

## 🔮 Future Work

- Integration with popular CAD software
- Support for more complex CAD operations
- Web-based interface for model generation
- Mobile deployment options
- Cross-domain adaptations (architectural, mechanical, etc.)

## 📄 License

[Add your license information here]

## 🙏 Acknowledgements

[Add acknowledgements, citations, or references to papers that inspired this work]

## 📬 Contact

[Add contact information]
