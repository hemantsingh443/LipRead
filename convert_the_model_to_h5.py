import os
import tensorflow as tf
from tensorflow.keras.models import Sequential

# Define the model architecture
def create_lipnet_model():
    from tensorflow.keras.layers import (
        Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, 
        TimeDistributed, Flatten
    )

    model = Sequential()

    # CNN layers
    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same', activation='relu'))
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

# Load the model
model = create_lipnet_model()

# Load TensorFlow checkpoint
checkpoint_path = os.path.join("..", "models", "checkpoint")
ckpt = tf.train.Checkpoint(model=model)

# Restore checkpoint
ckpt.restore(tf.train.latest_checkpoint(os.path.dirname(checkpoint_path))).expect_partial()
print("✅ Checkpoint loaded successfully.")

# Save weights in the correct format
model.save_weights("1-model_weights.weights.h5")
print("✅ Weights saved as 'model_weights.weights.h5'.") 
print("put the created file ////   model_weights.weights.h5   /////  in the models/")
