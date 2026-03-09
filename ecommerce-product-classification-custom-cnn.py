import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
#from example import model
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.optimizers import Adam
from tensorflow.python.ops.gen_dataset_ops import model_dataset

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
    # check the file is existed, if not, then skip it
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
# val doesn't need to be augmentate
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(train_pics, train_labels, batch_size=64)
val_generator = val_datagen.flow(val_pics, val_labels, batch_size=64)

# CNN model
from keras.layers import BatchNormalization, Activation

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam

model = Sequential()

# --- 第一組卷積層 ---
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(100, 100, 3)))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

# --- 第二組卷積層 ---
model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

# --- 第三組卷積層 (高階特徵) ---
model.add(Conv2D(128, (3, 3), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

# --- 全連接層 (Dense Layers) ---
model.add(Flatten())
model.add(Dense(512))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

# --- 輸出層 (9 個類別) ---
model.add(Dense(9, activation='softmax'))

# --- 編譯參數 (你最後成功的低學習率) ---
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

from keras.callbacks import EarlyStopping

# EarlyStopping callback, in case overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, start_from_epoch=5,verbose=1)

from sklearn.utils import class_weight
import numpy as np


y_train_indices = np.argmax(train_labels, axis=1)

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_indices),
    y=y_train_indices
)

class_weights_dict = dict(enumerate(weights))

print("The weight distribution is：", class_weights_dict)

# Train the model
history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=20,
                    callbacks=[early_stop],
                    #class_weight=class_weights_dict,
                    verbose=2 )



model.save('ecommerce_model.keras')
print("模型已儲存為 ecommerce_model.keras")

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

import numpy as np
from sklearn.metrics import classification_report

# 取得驗證集的所有預測值
y_pred = model.predict(val_pics)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(val_labels, axis=1)

# 顯示分類報告（精確率、召回率等）
print(classification_report(y_true, y_pred_classes, target_names=list(dict_label.keys())))

# 繪製混淆矩陣
cm = confusion_matrix(y_true, y_pred_classes)
labels_name = list(dict_label.keys())
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_name)
disp.plot(xticks_rotation=45, cmap='Blues', ax=plt.gca())
plt.title('Confusion Matrix')
plt.show()

score = model.evaluate(val_pics, val_labels)
print("Validation Accuracy =", score[1])