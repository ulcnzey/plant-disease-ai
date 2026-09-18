import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from tensorflow.keras.applications import MobileNetV3Small


# ============================================================
# AYARLAR
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

DATASET_DIR = "../dataset/PlantVillage"
MODEL_PATH = "../models/plant_disease_best.keras"
OUTPUT_MODEL = "../models/plant_disease_finetuned.keras"
OUTPUT_DIR = "../output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("🌱 PLANT DISEASE AI - FINE-TUNING")
print("=" * 70)


# ============================================================
# DATASET
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

print(f"\n🌿 Sınıf sayısı: {len(class_names)}")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
validation_ds = validation_ds.prefetch(AUTOTUNE)


# ============================================================
# MODELİ YÜKLE
# ============================================================

print("\n🧠 Mevcut model yükleniyor...")

model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Model yüklendi.")


# ============================================================
# MOBILENETV3SMALL TABANINI BUL
# ============================================================

base_model = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        if "mobilenetv3" in layer.name.lower():
            base_model = layer
            break

if base_model is None:
    raise RuntimeError("MobileNetV3Small tabanı bulunamadı!")

print(f"✅ Base model bulundu: {base_model.name}")


# ============================================================
# FINE-TUNING
# ============================================================

print("\n🔓 Fine-tuning başlıyor...")

base_model.trainable = True

# İlk katmanların çoğunu dondur.
# Son yaklaşık 30 katman öğrenebilir olacak.
for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    layer.trainable = True

# Batch Normalization katmanlarını dondur.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False


# ============================================================
# MODELİ COMPILE ET
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=1,
        min_lr=1e-7
    ),

    ModelCheckpoint(
        OUTPUT_MODEL,
        monitor="val_accuracy",
        save_best_only=True
    )
]


# ============================================================
# EĞİTİM
# ============================================================

print("\n🚀 Fine-tuning eğitimi başlıyor...\n")

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ============================================================
# MODELİ KAYDET
# ============================================================

model.save(OUTPUT_MODEL)

print("\n✅ Fine-tuned model kaydedildi:")
print(OUTPUT_MODEL)


# ============================================================
# ACCURACY GRAFİĞİ
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

plt.title("Fine-Tuning Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "fine_tuning_accuracy.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# LOSS GRAFİĞİ
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

plt.title("Fine-Tuning Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "fine_tuning_loss.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print("\n📊 Fine-tuning grafikleri kaydedildi.")

print("\n" + "=" * 70)
print("🎉 FINE-TUNING TAMAMLANDI!")
print("=" * 70)