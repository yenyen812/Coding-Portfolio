import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# the input
pic_size = (100, 100)
dict_label = {'BABY_PRODUCTS':0,
              'BEAUTY_HEALTH':1,
              'CLOTHING_ACCESSORIES_JEWELLERY':2,
              'ELECTRONICS':3,
              'GROCERY':4,
              'HOME_KITCHEN_TOOLS':5,
              'PET_SUPPLIES':6,
              'SPORTS_OUTDOOR':7,
              'HOBBY_ARTS_STATIONERY':8}

def load_images_from_folder(folder_path):
    class_images = {key: [] for key in dict_label}
    # check the file is exist, if not, then skip it
    for folder in glob.glob(os.path.join(folder_path, '*')):
        label = os.path.basename(folder)
        if label not in dict_label:
            continue

        for filename in os.listdir(folder):
            try:
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, dsize=pic_size) #resize all img to 100x100
                    class_images[label].append(img)
            # check if there is any img cannot be load and pass
            except:
                print("Cannot read from file " + img_path)
                pass
    pics = []
    labels = []
    for label_name, images in class_images.items():
        pics.extend(images)
        labels.extend([dict_label[label_name]] * len(images))
        # Normalization
    return np.array(pics, dtype='float32') / 255.0, to_categorical(np.array(labels), num_classes=9)


train_folder = "/Users/ykk/Downloads/電商/train"
val_folder = "/Users/ykk/Downloads/電商/val"

train_pics, train_labels = load_images_from_folder(train_folder)
val_pics, val_labels = load_images_from_folder(val_folder)

# Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# val doesn't need to be augment
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(train_pics, train_labels, batch_size=64)
val_generator = val_datagen.flow(val_pics, val_labels, batch_size=64)

# CNN model
model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(5,5), padding='same', input_shape=(100,100,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters=20, kernel_size=(5,5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=9, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping

# EarlyStopping callback, in case overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

# Train the model
history = model.fit(
    x=train_pics,
    y=train_labels,
    validation_data=(val_pics, val_labels),
    epochs=20,
    batch_size=64,
    callbacks=[early_stop],
    verbose=2
)

# plot the figure
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

# Prediction and Confusion Matrix
# y_pred = model.predict(val_pics)
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_true = np.argmax(val_labels, axis=1)
#
# cm = confusion_matrix(y_true, y_pred_classes)
# labels_name = list(dict_label.keys())
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_name)
# plt.figure(figsize=(10, 10))
# disp.plot(xticks_rotation=45, cmap='Blues')
# plt.title('Confusion Matrix')
# plt.show()

score = model.evaluate(val_pics, val_labels)
print("Validation Accuracy =", score[1])