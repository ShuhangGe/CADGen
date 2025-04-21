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

## Detailed Model Architectures

### Base Model (Views2Points)
The base model takes multi-view images and generates point clouds representing the 3D shape:
- **Backbone**: ResNet-based feature extraction from each view
- **Fusion Module**: Multi-view feature integration
- **Decoder**: Transformer architecture that outputs 3D points and CAD command sequences

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

## Dataset Format

The project expects data in the following format:

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

## Performance Metrics

The framework evaluates performance using:

1. **Chamfer Distance (CD)**: Measures the similarity between generated and ground truth point clouds
2. **Command Accuracy**: Percentage of correctly predicted CAD commands
3. **Parameter Error**: Mean squared error between predicted and ground truth parameters
4. **Visual Quality**: Qualitative assessment of the generated 3D models

Benchmark results on test set:
| Model | CD ↓ | Cmd Acc ↑ | Param MSE ↓ |
|-------|------|-----------|-------------|
| Base  | -    | -         | -           |
| Deformable | - | -       | -           |
| Autoregressive | - | -   | -           |

## Inference Examples

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

## Troubleshooting

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

## Contributing

Contributions to CADGen are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the project's style guidelines and includes appropriate tests.

## Future Work

- Integration with popular CAD software
- Support for more complex CAD operations
- Web-based interface for model generation
- Mobile deployment options
- Cross-domain adaptations (architectural, mechanical, etc.)

## License

[Add your license information here]

## Acknowledgements

[Add acknowledgements, citations, or references to papers that inspired this work]

## Contact

[Add contact information]
