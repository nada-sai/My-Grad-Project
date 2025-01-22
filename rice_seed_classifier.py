import os
import shutil
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Path to the dataset directory containing the images
dataset_dir = 'C:/Users/DELL/Desktop/Rice_Image_Dataset'

# List of class names
classes = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Split ratio for train and validation sets (80% train, 20% validation)
split_ratio = 0.8

# Directories for training and validation datasets
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'validation')

# Create train and validation directories if they do not exist
for class_name in classes:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

# Loop through each class and split the images
for class_name in classes:
    # Path to the current class folder
    class_folder = os.path.join(dataset_dir, class_name)
    
    # Get the list of all image files in the current class
    image_files = [f for f in os.listdir(class_folder) if f.endswith('.jpg') or f.endswith('.png')]

    # Shuffle the image list to randomize the split
    random.shuffle(image_files)

    # Calculate the split index
    split_index = int(len(image_files) * split_ratio)

    # Split the image files into train and validation sets
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # Move the images to the respective directories
    for file in train_files:
        shutil.move(os.path.join(class_folder, file), os.path.join(train_dir, class_name, file))
    
    for file in val_files:
        shutil.move(os.path.join(class_folder, file), os.path.join(val_dir, class_name, file))

print("Dataset has been successfully split into train and validation sets.")

# Data augmentation and preprocessing for training and validation sets
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=40, 
    width_shift_range=0.2,
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.2,
    horizontal_flip=True, 
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, 
    target_size=(150, 150),
    batch_size=32, 
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir, 
    target_size=(150, 150),
    batch_size=32, 
    class_mode='categorical'
)

# Model definition (simple CNN model)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5 classes for rice types
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model for 10 epochs
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# Plot training and validation loss and accuracy
plt.figure(figsize=(12, 4))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Saving the trained model
model.save('rice_seed_classifier.h5')

print("Model training completed and saved successfully.")
