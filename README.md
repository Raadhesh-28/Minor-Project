# Automated Classification of Indian Traditional Painting Styles using a Hybrid Deep Learning Approach

## Project Overview
This repository contains the source code for classifying various Indian traditional painting styles using a robust Hybrid Deep Learning approach. The project combines state-of-the-art Convolutional Neural Networks (CNN) with custom handcrafted feature extraction (color, texture, shape, symmetry) to accurately identify the unique stylistic and structural traits of regional Indian art forms.

## Dataset
The project utilizes the [Indian Paintings Dataset from Kaggle](https://www.kaggle.com/datasets/ajg117/indian-paintings-dataset).
It classifies 8 distinct Indian traditional painting styles:
- Warli Painting
- Pichwai Painting
- Mandana Art Drawing
- Madhubani Painting
- Kerala Mural
- Kangra Painting
- Kalighat Painting
- Gond Painting

## Model Architecture (Hybrid Model)
Our classification architecture uses a Hybrid Feature Extractor which maps images into a 519-dimensional feature space, consisting of:
1. **CNN Embeddings (512-dim):** We leverage a pre-trained **ResNet18** model as a backbone (with frozen weights) to extract high-level semantic and spatial representations.
2. **Handcrafted Features (7-dim):** Traditional visual heuristics are extracted to capture structural elements:
   - `mean_r`, `mean_g`, `mean_b` (Color Distribution)
   - `color_variance`
   - `edge_density` (Shape/Structure)
   - `symmetry_score` (Structural Balance)
   - `texture_entropy` (Texture Variations)

These features are fused and passed through a custom Multi-Layer Perceptron (MLP) mapping `519 -> 256 -> 8` classes.

## Repository Structure
```
├── configs/            # Configuration files (config.yaml)
├── data/               # Metadata mappings and extracted features CSV
├── models/             # Saved checkpoints (.pth)
├── src/                # Core implementation
│   ├── features/       # Feature extraction logic (color, shape, texture)
│   ├── models/         # ResNet CNN and Hybrid Model structures
│   ├── training/       # Training and evaluation loops
│   └── utils/          # Helper modules
├── Minor_Project (3).ipynb # End-to-end execution and experimentation notebook
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Raadhesh-28/Automated-Classification-of-Indian-Traditional-Painting-styles-using-a-Hybrid-Deep-learning-approach.git
   cd Automated-Classification-of-Indian-Traditional-Painting-styles-using-a-Hybrid-Deep-learning-approach
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the Kaggle Dataset:
   Ensure you have configured your `kaggle.json` inside `~/.kaggle/` and download the dataset:
   ```bash
   kaggle datasets download -d ajg117/indian-paintings-dataset
   unzip indian-paintings-dataset.zip -d data/raw
   ```

## Usage and Training
The execution flow generally consists of extracting handcrafted features, then training the model. You can follow the steps mapped out in `Minor_Project (3).ipynb` or execute the Python scripts manually:

1. **Feature Extraction**:
   ```bash
   python -m src.features.extract_features
   ```
   *This precomputes and normalizes the structural parameters and outputs them into a CSV file.*

2. **Train the Hybrid Model**:
   ```bash
   python -m src.models.hybrid_model
   ```
   *This initializes the memory mapping, fuses the ResNet18 embeddings with the handcrafted CSV parameters, and outputs the optimal best-validation model to `models/hybrid_classifier.pth`.*

3. **Train standard CNN Model (Baseline)**:
   ```bash
   python -m src.training.train
   ```

## Evaluation and Results
The dataset is split into an 80/20 train/validation ratio. The evaluation computes the validation accuracy and standard cross-entropy loss dynamically per epoch.
Metrics and visualizations such as **Confusion Matrices** and **PCA** space scatter plots (projecting 519-dimensions down to 2 principal components) are available in the accompanying Jupyter Notebook (`Minor_Project (3).ipynb`).

## Future Work
- Evaluating other heavier backbone architectures like ResNet50 or EfficientNet.
- Deploying the model using a FastAPI/Flask inference endpoint.
- Expanding the dataset to include variations and rarer styles of regional artwork.