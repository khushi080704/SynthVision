# SynthVision: Generative Image Synthesis & Typology Classification

An applied computer vision framework utilizing StyleGAN2-ADA and ResNet-18 to handle class sparsity in structural imagery and evaluate classification benchmarks across roof typologies (RCC, Metal Sheet, Assam Types, Vacant).

## Architecture Highlights
- Preprocessing: Custom Lanczos aspect-ratio preservation and pad formatting to 256x256 resolution.
- Conditional Generation: Formatted class-conditional manifests (dataset.json) to train StyleGAN2-ADA under limited data constraints using Adaptive Discriminator Augmentation (ADA).
- Transfer Learning Classifier: ResNet-18 fine-tuning with frozen backbone representations, evaluated using multi-class confusion matrices and classification reports.

## Repository Structure
- dataset_preprocess.py: Image padding & StyleGAN JSON packaging
- train_classifier.py: ResNet-18 fine-tuning & evaluation script
- requirements.txt: Environment dependencies
- .gitignore: Ignored large files and checkpoints
- README.md: Project overview and setup instructions

## Running the Pipeline

1. Install dependencies:
pip install -r requirements.txt

2. Run preprocessing:
python dataset_preprocess.py

3. Train Classifier:
python train_classifier.py --data ./data