import os
import random
import shutil
from pathlib import Path

base_dir = Path("dogs_vs_cats_small")
train_dir = base_dir / "train"
val_dir = base_dir / "val"


for split in ["train", "val"]:
    for cls in ["cat", "dog"]: 
        (base_dir / split / cls).mkdir(parents=True, exist_ok=True)

cat_files = list((base_dir / "Cat_Original").glob("*.jpg"))   
dog_files = list((base_dir / "Dog_Original").glob("*.jpg")) 


random.shuffle(cat_files)
random.shuffle(dog_files)

train_cats = cat_files[:500]
train_dogs = dog_files[:500]

val_cats = cat_files[500:600]
val_dogs = dog_files[500:600]


for f in train_cats:
    shutil.move(f, base_dir / "train" / "cat" / f.name) 
for f in train_dogs:
    shutil.move(f, base_dir / "train" / "dog" / f.name)
for f in val_cats:
    shutil.move(f, base_dir / "val" / "cat" / f.name)
for f in val_dogs:
    shutil.move(f, base_dir / "val" / "dog" / f.name)

print(f"Training set: {len(train_cats)} images of cats, {len(train_dogs)} images of dogs")
print(f"Validation set: {len(val_cats)} images of cats, {len(val_dogs)} images of dogs")