import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Input settings
pic_size = (100, 100)
dict_label = {
    'BABY_PRODUCTS': 0,
    'BEAUTY_HEALTH': 1,
    'CLOTHING_ACCESSORIES_JEWELLERY': 2,
    'ELECTRONICS': 3,
    'GROCERY': 4,
    'HOME_KITCHEN_TOOLS': 5,
    'PET_SUPPLIES': 6,
    'SPORTS_OUTDOOR': 7,
    'HOBBY_ARTS_STATIONERY': 8
}

def load_images_from_folder(folder_path):
    class_images = {key: [] for key in dict_label}
    for folder in glob.glob(os.path.join(folder_path, '*')):
        label = os.path.basename(folder)
        if label not in dict_label:
            continue

        for filename in os.listdir(folder):
            try:
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, dsize=pic_size)
                    class_images[label].append(img)
            except:
                print("Cannot read from file " + img_path)
                pass

    pics = []
    labels = []
    for label_name, images in class_images.items():
        pics.extend(images)
        labels.extend([dict_label[label_name]] * len(images))

    return np.array(pics, dtype='float32') / 255.0, to_categorical(np.array(labels), num_classes=9)


train_folder = "/Users/ykk/Downloads/ecommerce/train"
val_folder = "/Users/ykk/Downloads/ecommerce/val"

train_pics, train_labels = load_images_from_folder(train_folder)
val_pics, val_labels = load_images_from_folder(val_folder)

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()

val_generator = val_datagen.flow(val_pics, val_labels, batch_size=64, shuffle=False)

# Undersampling
def undersample_to_target(train_pics, train_labels, target_count=600):
    y_train_indices = np.argmax(train_labels, axis=1)
    class_indices = {i: np.where(y_train_indices == i)[0] for i in np.unique(y_train_indices)}

    sampled_indices = []
    for cls, idxs in class_indices.items():
        if len(idxs) > target_count:
            sampled = np.random.choice(idxs, size=target_count, replace=False)
        else:
            sampled = idxs
        sampled_indices.extend(sampled)

    sampled_indices = np.array(sampled_indices)
    np.random.shuffle(sampled_indices)

    return train_pics[sampled_indices], train_labels[sampled_indices]

# CNN model
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(100, 100, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(9, activation='softmax'))

# Compile settings
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    start_from_epoch=5,
    verbose=1
)

# Check original class distribution
y_train_indices = np.argmax(train_labels, axis=1)
unique, counts = np.unique(y_train_indices, return_counts=True)
print("Original training class distribution:", dict(zip(unique, counts)))

# Apply undersampling
X_under, y_under = undersample_to_target(train_pics, train_labels, target_count=600)

# Check class distribution after undersampling
y_under_indices = np.argmax(y_under, axis=1)
unique_under, counts_under = np.unique(y_under_indices, return_counts=True)
print("Undersampled class distribution:", dict(zip(unique_under, counts_under)))

# Build training generator using undersampled data
train_generator = train_datagen.flow(X_under, y_under, batch_size=64, shuffle=True)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop],
    verbose=2
)

model.save('ecommerce_model.keras')
print("Model saved as ecommerce_model.keras")

# Training curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Get predictions
y_pred = model.predict(val_pics)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(val_labels, axis=1)

# classification report
print(classification_report(y_true, y_pred_classes, target_names=list(dict_label.keys())))

# confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
labels_name = list(dict_label.keys())
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_name)
disp.plot(xticks_rotation=45, cmap='Blues', ax=plt.gca())
plt.title('Confusion Matrix')
plt.show()

score = model.evaluate(val_pics, val_labels)
print("Validation Accuracy =", score[1])