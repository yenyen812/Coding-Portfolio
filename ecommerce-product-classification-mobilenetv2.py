import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score
from sklearn.utils import class_weight
from keras.applications import MobileNetV2
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import EarlyStopping

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
    class_counts = {key: 0 for key in dict_label}
    for folder in glob.glob(os.path.join(folder_path, '*')):
        label = os.path.basename(folder)
        if label not in dict_label:
            continue
        print(f"{label} image files reading ...")
        count = 0
        for filename in os.listdir(folder):
            try:
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, dsize=pic_size)
                    class_images[label].append(img)
                    count += 1
            except:
                print("Cannot read from file " + img_path)
                pass
        class_counts[label] = count
    pics = []
    labels = []
    for label_name, images in class_images.items():
        pics.extend(images)
        labels.extend([dict_label[label_name]] * len(images))
    return np.array(pics, dtype='float32') / 255.0, to_categorical(np.array(labels), num_classes=9), class_counts


train_folder = "/Users/ykk/Downloads/電商/train"
val_folder = "/Users/ykk/Downloads/電商/val"

train_pics, train_labels, train_class_counts = load_images_from_folder(train_folder)
val_pics, val_labels, val_class_counts = load_images_from_folder(val_folder)

# 計算類別權重
class_labels = list(dict_label.values())
class_names = list(dict_label.keys())
train_label_indices = np.argmax(train_labels, axis=1)
calculated_class_weight = class_weight.compute_class_weight('balanced',
                                                            classes=np.unique(train_label_indices),
                                                            y=train_label_indices)
class_weight_dict = dict(zip(np.unique(train_label_indices), calculated_class_weight))
print("Class Weights:", class_weight_dict)

# Define data augmentation techniques
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range = 0.2,
)
#
val_datagen = ImageDataGenerator(rescale=1./255)
#
train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

IMG_SHAPE = (100, 100, 3)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

predictions = Dense(9, activation='softmax')(x)


model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

history = model.fit(x=train_pics, y=train_labels, validation_data=(val_pics, val_labels), epochs=100, batch_size=64, verbose=2, callbacks=[early_stopping], class_weight=class_weight_dict)

# Final plot
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

# Cinfusion Matrix
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

# Calculate the precision rate
# precision = precision_score(y_true, y_pred_classes, labels=np.arange(len(labels_name)), average=None)
# for i, label in enumerate(labels_name):
#     print(f"Category '{label}' _Precision rate: {precision[i]:.4f}")