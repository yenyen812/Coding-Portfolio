import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

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

# Focal loss with reciprocal
class_counts = np.sum(train_labels, axis=0)
print("class_counts:", class_counts)

alpha = 1.0 / class_counts
alpha = alpha / np.sum(alpha) * len(class_counts)
alpha = tf.constant(alpha, dtype=tf.float32)

print("alpha:", alpha.numpy())

train_datagen = ImageDataGenerator(
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(train_pics, train_labels, batch_size=64, shuffle=True)
val_generator = val_datagen.flow(val_pics, val_labels, batch_size=64, shuffle=False)

def categorical_focal_loss(gamma=2.0, alpha=None):
    def focal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1 - y_pred, gamma)

        if alpha is not None:
            alpha_weight = y_true * alpha
            loss = alpha_weight * focal_weight * cross_entropy
        else:
            loss = focal_weight * cross_entropy

        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return focal_loss

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

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=categorical_focal_loss(gamma=2.0, alpha=alpha), #focal loss
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

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop],
    verbose=2
)

model.save('ecommerce_model.keras')

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

y_pred = model.predict(val_pics)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(val_labels, axis=1)

print(classification_report(y_true, y_pred_classes, target_names=list(dict_label.keys())))

cm = confusion_matrix(y_true, y_pred_classes)
labels_name = list(dict_label.keys())
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_name)
disp.plot(xticks_rotation=45, cmap='Blues', ax=plt.gca())
plt.title('Confusion Matrix')
plt.show()

score = model.evaluate(val_pics, val_labels)
print("Validation Accuracy =", score[1])