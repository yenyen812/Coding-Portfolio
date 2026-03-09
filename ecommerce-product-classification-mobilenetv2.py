import os
import glob
import cv2
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# input
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
                    img = cv2.resize(img, dsize=pic_size)  # resize all images to 100x100
                    class_images[label].append(img)
            # check if there is any image that cannot be loaded and skip it
            except:
                print("Cannot read from file " + img_path)
                pass
    pics = []
    labels = []
    for label_name, images in class_images.items():
        pics.extend(images)
        labels.extend([dict_label[label_name]] * len(images))
        # normalization
    return np.array(pics, dtype='float32') / 255.0, to_categorical(np.array(labels), num_classes=9)


train_folder = "/Users/ykk/Downloads/ecommerce/train"
val_folder = "/Users/ykk/Downloads/ecommerce/val"

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

# validation data does not need augmentation
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(train_pics, train_labels, batch_size=64)
val_generator = val_datagen.flow(val_pics, val_labels, batch_size=64)

from keras.layers import BatchNormalization, Activation
from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam

# 1. Load the pretrained model (without the final fully connected layers)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# 2. Freeze the base model weights first
base_model.trainable = True

# 3. Build your own classifier
x = base_model.output
x = GlobalAveragePooling2D()(x)  # convert feature maps into a 1D vector
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)              # reduce overfitting
predictions = Dense(9, activation='softmax')(x)  # output layer (9 classes)

# 4. Combine into a new model
model = Model(inputs=base_model.input, outputs=predictions)

# 5. Compile the model
model.compile(optimizer=Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

from keras.callbacks import EarlyStopping

# EarlyStopping callback in case of overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

from sklearn.utils import class_weight
import numpy as np

# 1. Get all training label indices (assuming labels are one-hot encoded)
y_train_indices = np.argmax(train_labels, axis=1)

# 2. Compute class weights: fewer samples get higher weights
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_indices),
    y=y_train_indices
)

# 3. Convert to the dictionary format required by Keras {class_index: weight}
class_weights_dict = dict(enumerate(weights))

print("Computed class weight distribution:", class_weights_dict)

history = model.fit(
    train_generator,  # use generator here to apply rotation, zoom, and other augmentations
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop],
    class_weight=class_weights_dict,
    verbose=2
)

model.save('ecommerce_model.keras')
print("Model saved as ecommerce_model.keras")

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

# Get all predictions on the validation set
y_pred = model.predict(val_pics)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(val_labels, axis=1)

# Show classification report (precision, recall, etc.)
print(classification_report(y_true, y_pred_classes, target_names=list(dict_label.keys())))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
labels_name = list(dict_label.keys())
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_name)
disp.plot(xticks_rotation=45, cmap='Blues', ax=plt.gca())
plt.title('Confusion Matrix')
plt.show()

score = model.evaluate(val_pics, val_labels)
print("Validation Accuracy =", score[1])