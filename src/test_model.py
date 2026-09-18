import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ============================================================
# AYARLAR
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

DATASET_DIR = "../dataset/PlantVillage"
MODEL_PATH = "../models/plant_disease_best.keras"
OUTPUT_DIR = "../output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("🧪 PLANT DISEASE AI - MODEL TESTİ")
print("=" * 70)

# ============================================================
# TEST DATASET
# ============================================================

print("\n📂 Test dataset hazırlanıyor...")

# Dataset'i aynı seed ile karıştırarak yaklaşık %10 test,
# %10 validation ve %80 training olarak ayırıyoruz.
test_full = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = test_full.class_names

print(f"🌿 Sınıf sayısı: {len(class_names)}")

# Validation'ın yarısını test olarak kullanıyoruz.
total_batches = tf.data.experimental.cardinality(test_full).numpy()
test_batches = total_batches // 2

test_ds = test_full.take(test_batches)

print(f"🧪 Test batch sayısı: {test_batches}")

# ============================================================
# MODEL
# ============================================================

print("\n🧠 Model yükleniyor...")

model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Model yüklendi.")

# ============================================================
# TEST ACCURACY
# ============================================================

print("\n🚀 Test başlıyor...\n")

test_loss, test_accuracy = model.evaluate(test_ds)

print("\n" + "=" * 70)
print("📊 TEST SONUÇLARI")
print("=" * 70)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

# ============================================================
# TAHMİNLER
# ============================================================

print("\n🔍 Tahminler oluşturuluyor...")

y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)

    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print("\n📋 CLASSIFICATION REPORT")
print("=" * 70)
print(report)

with open(
    os.path.join(OUTPUT_DIR, "classification_report.txt"),
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(18, 16))

plt.imshow(cm)

plt.title("Plant Disease Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=90,
    fontsize=7
)

plt.yticks(
    range(len(class_names)),
    class_names,
    fontsize=7
)

plt.colorbar()

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\n📊 Confusion matrix kaydedildi.")

print("\n" + "=" * 70)
print("🎉 TEST TAMAMLANDI!")
print("=" * 70)