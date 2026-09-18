import tensorflow as tf
from pathlib import Path

# Model yolu
model_path = Path("models/plant_disease_finetuned.keras")

# Çıktı yolu
output_path = Path("assets/model.tflite")

print("🌱 Model yükleniyor...")
print(f"Model: {model_path}")

# Keras modelini yükle
model = tf.keras.models.load_model(model_path)

print("✅ Model başarıyla yüklendi!")
print("🔄 TFLite dönüşümü başlıyor...")

# TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# İlk aşamada optimizasyon kullanmıyoruz.
# Böylece float32 model elde ediyoruz.
tflite_model = converter.convert()

# Dosyaya kaydet
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "wb") as f:
    f.write(tflite_model)

print("✅ TFLite modeli oluşturuldu!")
print(f"📱 Dosya: {output_path}")
print(f"📦 Boyut: {output_path.stat().st_size / (1024 * 1024):.2f} MB")