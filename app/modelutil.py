import os
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Input

def create_lipnet_model():
    from tensorflow.keras.layers import (
        Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, 
        TimeDistributed, Flatten
    )

    model = Sequential()

    # CNN layers
    model.add(Input(shape=(75, 46, 140, 1)))  # Define input shape separately
    model.add(Conv3D(128, 3, padding='same', activation='relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same', activation='relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same', activation='relu'))
    model.add(MaxPool3D((1, 2, 2)))

    # Flatten the features per frame
    model.add(TimeDistributed(Flatten()))

    # LSTM layers
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    # Output layer with softmax activation
    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    return model

# Load the model architecture
model = create_lipnet_model()

# Path to the saved weights
import os

weights_path = os.path.join("..", "models", "model_weights.weights.h5")
absolute_path = os.path.abspath(weights_path)

if not os.path.exists(absolute_path):
    raise FileNotFoundError(f"❌ Weights file not found at: {absolute_path}")

try:
    model.load_weights(absolute_path)
    print("✅ Model weights loaded successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Error loading model weights: {e}")


