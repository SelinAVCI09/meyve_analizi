import os
import shutil
import random

# Veri seti dizini
dataset_dir = '/Users/selinavci/Desktop/meyve_analizi/Fruits_Vegetables_Dataset(12000)/'

# Train ve test için klasör yolları
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')

# Fruit ve vegetable dizinleri
fruits_dir = os.path.join(dataset_dir, 'Fruits')
vegetables_dir = os.path.join(dataset_dir, 'Vegetables')

# Helper function to split data
def split_data(source_dir, train_dir, test_dir, split_ratio=0.8):
    # Klasörler arası geçiş yaparak her kategori için veri ayırma
    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)
        
        if os.path.isdir(category_path):
            # Train ve Test için alt klasör oluştur
            train_category_path = os.path.join(train_dir, category)
            test_category_path = os.path.join(test_dir, category)
            os.makedirs(train_category_path, exist_ok=True)
            os.makedirs(test_category_path, exist_ok=True)
            
            # Kategorideki tüm resimleri al
            images = os.listdir(category_path)
            random.shuffle(images)
            
            # Eğitim ve test için resim sayısını ayarlıyoruz
            split_index = int(len(images) * split_ratio)
            train_images = images[:split_index]
            test_images = images[split_index:]
            
            # Resimleri train ve test klasörlerine kopyala
            for image in train_images:
                shutil.copy(os.path.join(category_path, image), os.path.join(train_category_path, image))
            for image in test_images:
                shutil.copy(os.path.join(category_path, image), os.path.join(test_category_path, image))

# Fruits ve Vegetables dizinlerini ayır
split_data(fruits_dir, train_dir, test_dir)
split_data(vegetables_dir, train_dir, test_dir)

print("Veri başarıyla train ve test olarak ayrıldı.")
