import os
import sys
import tensorflow as tf

# Allow importing model.py from the same folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import model

DATA_DIR = "data/PlantVillage/raw/color"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Load training data
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load validation data
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Performance optimization
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Train EfficientNet model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

# Save final model
model.save("results/plant_disease_model.keras")

print("Training completed!")
print("Model saved to results/plant_disease_model.keras")