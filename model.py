import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def build_and_train_model(landmarks, labels):
    X_train, X_test, y_train, y_test = train_test_split(landmarks, labels, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(landmarks.shape[1], landmarks.shape[2])),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(np.unique(labels)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

    return model
