import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# ============================================================
# 1. AYARLAR
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

DATASET_DIR = "../dataset/PlantVillage"
MODEL_DIR = "../models"
OUTPUT_DIR = "../output"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("🌱 PLANT DISEASE AI - MODEL EĞİTİMİ")
print("=" * 70)

# ============================================================
# 2. DATASET
# ============================================================

print("\n📂 Dataset yükleniyor...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

print(f"\n🌿 Sınıf sayısı: {num_classes}")
print(f"🖼️ Sınıflar: {class_names}")

# ============================================================
# 3. PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
validation_ds = validation_ds.prefetch(AUTOTUNE)

# ============================================================
# 4. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
], name="data_augmentation")

# ============================================================
# 5. MOBILENETV3SMALL
# ============================================================

print("\n🧠 MobileNetV3Small hazırlanıyor...")

base_model = MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
    include_preprocessing=True
)

base_model.trainable = False

# ============================================================
# 6. MODEL
# ============================================================

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 7. CALLBACKS
# ============================================================

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6
    ),

    ModelCheckpoint(
        os.path.join(MODEL_DIR, "plant_disease_best.keras"),
        monitor="val_accuracy",
        save_best_only=True
    )
]

# ============================================================
# 8. EĞİTİM
# ============================================================

print("\n🚀 MODEL EĞİTİMİ BAŞLIYOR...")
print("Bu işlem bilgisayarına göre biraz zaman alabilir.\n")

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ============================================================
# 9. MODELİ KAYDET
# ============================================================

model.save(
    os.path.join(MODEL_DIR, "plant_disease_model.keras")
)

print("\n✅ Model kaydedildi.")

# ============================================================
# 10. ACCURACY GRAFİĞİ
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Plant Disease Classification Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(OUTPUT_DIR, "accuracy.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 11. LOSS GRAFİĞİ
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Plant Disease Classification Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(OUTPUT_DIR, "loss.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("📊 Grafikler output klasörüne kaydedildi.")

print("\n" + "=" * 70)
print("🎉 İLK EĞİTİM TAMAMLANDI!")
print("=" * 70)