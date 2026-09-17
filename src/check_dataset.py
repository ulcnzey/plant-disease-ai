import os

DATASET_DIR = "../dataset/PlantVillage"

print("=" * 60)
print("🌱 PLANTVILLAGE DATASET KONTROLÜ")
print("=" * 60)

if not os.path.exists(DATASET_DIR):
    print("❌ Dataset bulunamadı!")
    print(f"Aranan konum: {DATASET_DIR}")
    exit()

classes = sorted(
    [
        folder
        for folder in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, folder))
    ]
)

print("\n✅ Dataset bulundu.")
print(f"📁 Toplam sınıf sayısı: {len(classes)}")

print("\n📋 SINIFLAR")
print("-" * 60)

total_images = 0

for index, class_name in enumerate(classes):
    class_path = os.path.join(DATASET_DIR, class_name)

    images = [
        file
        for file in os.listdir(class_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_count = len(images)
    total_images += image_count

    print(
        f"{index:02d} | "
        f"{class_name:<40} | "
        f"{image_count} görüntü"
    )

print("-" * 60)

print(f"\n🖼️ Toplam görüntü: {total_images}")
print(f"🌿 Toplam sınıf: {len(classes)}")

print("\n✅ Dataset kontrolü tamamlandı.")