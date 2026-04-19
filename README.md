# Dogs vs Cats Image Classification with Transfer Learning

This project implements a binary image classification system to distinguish between dog and cat images using transfer learning techniques. The model compares performance between fine-tuning pre-trained ResNet models and training from scratch.

---

## 1. Experimental results

- Fine-tuned model: Validation accuracy = **0.8650** (86.50%) 
- Training from scratch: Validation accuracy = **0.5000** (50.00%) 
- ✅ Fine-tuning is significantly superior to training from scratch, verifying the strong generalization ability of pre-trained models on small datasets

---

## 2. Project structure

- `Cat_Original` file folder
- `Dog_Original` file folder  
Original dataset

- `train` file folder           
Contains `cat` and `dog` subfolders, each with 500 randomly selected images for training (cat photos and dog photos respectively)

- `val` file folder           
Contains `cat` and `dog` subfolders, each with 100 randomly selected images for validation (cat photos and dog photos respectively)

- `split_data.py`           
Utility script for splitting dataset into train/validation sets

- `requirements.txt`           
List of required dependencies

- `main.py`           
Main training script implementing fine-tuning vs training from scratch

- `README.md`
Project description document

- `accuracy_comparison.png`           
- `sample_predictions.png`           
Visualization results

---

## 3. Environment configuration

### 3.1 Create a virtual environment (recommended)
```bash
python -m venv venv
.venv\Scripts\activate
```

### 3.2 Install dependency packages
```bash
pip install -r requirements.txt
```

---

## 4. Program running

### 4.1 Data sampling

```bash
python split_data.py
```

The script will automatically:

1. Randomly select 500 photos of kittens and 500 photos of puppies from the train folder as test samples.
2. Randomly select 100 photos of kittens and 100 photos of puppies from the train folder as validation samples.

### 4.2 Core experimental code

```bash
python main.py
```

The script will automatically:

1. Generate accuracy_comparison.png           
2. Generate sample_predictions.png  
