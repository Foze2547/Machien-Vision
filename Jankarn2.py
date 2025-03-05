import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import cv2
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Set path
DATA_PATH = r"D:\\Dataset_for_development\\Test"
CSV_PATH = r"D:\\Dataset_for_development\\mix.csv"

# Load dataset
df = pd.read_csv(CSV_PATH)

# Image processing function
def load_and_preprocess_image(image_name):
    img_path = os.path.join(DATA_PATH, image_name)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image {image_name} not found at {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # Resize to 224x224 for CNN
    img = img / 255.0  # Normalize to 0-1
    return img

# Load images
X1 = np.array([load_and_preprocess_image(img) for img in df['Image 1']])
X2 = np.array([load_and_preprocess_image(img) for img in df['Image 2']])
y_classification = np.array(df['Winner']) - 1  # Convert 1,2 to 0,1

# Split dataset
X1_train, X1_test, X2_train, X2_test, y_class_train, y_class_test = train_test_split(
    X1, X2, y_classification, test_size=0.2, random_state=42)

# Define CNN base model
def create_base_cnn():
    input_layer = Input(shape=(224, 224, 3), name='input_layer')
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    return Model(input_layer, x)

# Create twin networks
base_cnn = create_base_cnn()
input_1 = Input(shape=(224, 224, 3), name='input_1')
input_2 = Input(shape=(224, 224, 3), name='input_2')

encoded_1 = base_cnn(input_1)
encoded_2 = base_cnn(input_2)

merged = Concatenate()([encoded_1, encoded_2])

# Classification head
class_output = Dense(1, activation='sigmoid', name='class_output')(merged)

# Build model
model = Model(inputs=[input_1, input_2], outputs=class_output)
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Train model
model.fit([X1_train, X2_train], y_class_train,
          validation_data=([X1_test, X2_test], y_class_test),
          epochs=100, batch_size=64)

model.save('trained_model2.h5')

# Evaluate model
loss, accuracy = model.evaluate([X1_test, X2_test], y_class_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
