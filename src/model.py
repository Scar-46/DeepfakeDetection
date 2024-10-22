import os
import matplotlib.pyplot as plt
import numpy as np

# Tensor libraries
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# Custom libraries
import config.configuration as config

# *************** Split Dataset ***********************
data_gen = ImageDataGenerator(rescale=1.0/255)
target_size = (config.TARGET_SIZE, config.TARGET_SIZE)

train_generator = data_gen.flow_from_directory(
    config.TRAIN_DIR,
    target_size=target_size,
    batch_size=config.BATCH_SIZE,
    color_mode='grayscale',
    class_mode='binary'
)

val_generator = data_gen.flow_from_directory(
    config.VAL_DIR,
    target_size=target_size,
    batch_size=config.BATCH_SIZE,
    color_mode='grayscale',
    class_mode='binary'
)

test_generator = data_gen.flow_from_directory(
    config.TEST_DIR,
    target_size=target_size,
    batch_size=config.BATCH_SIZE,
    color_mode='grayscale',
    class_mode='binary',
    shuffle=False
)

# *************** Model Configuration ***********************
# Use the correct input shape for grayscale images
base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(config.TARGET_SIZE, config.TARGET_SIZE, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # You can customize this layer size
predictions = Dense(1, activation='sigmoid')(x)  # Use 'sigmoid' for binary classification

model = Model(inputs=base_model.input, outputs=predictions)

# Change the loss function to binary_crossentropy for binary classification
model.compile(optimizer="adam",
              loss="binary_crossentropy",  # Use binary_crossentropy for binary classification
              metrics=["accuracy"])

# model.summary()

history = model.fit(train_generator, validation_data=val_generator, epochs=config.N_EPOCHS, verbose=1)

# *************** Plotting Results ***********************
# Plotting accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# *************** Model Performance ***********************
predictions = model.predict(test_generator, verbose=1)
predicted_percentages = predictions * 100
predicted_classes = np.where(predicted_percentages > 50, 1, 0)

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print(classification_report(true_classes, predicted_classes, target_names=class_labels))