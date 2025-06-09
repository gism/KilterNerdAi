import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Load dataset from JSON file
with open("training_data_extended.json", "r") as f:
    data = json.load(f)

# Extract features and labels
frames = [item["frame"] for item in data]
angles = [item["angle"] for item in data]
grade = [item["grade"] for item in data]

# Tokenize the frame strings
tokenizer = Tokenizer()
tokenizer.fit_on_texts(frames)
frame_sequences = tokenizer.texts_to_sequences(frames)
frame_sequences = pad_sequences(frame_sequences, padding='post')

# Convert angle and difficulty to numpy arrays
angles = np.array(angles).reshape(-1, 1).astype(np.float32)
grade = np.array(grade).reshape(-1, 1).astype(np.float32)

# Split into training and test sets
X_frames_train, X_frames_test, X_angles_train, X_angles_test, y_train, y_test = train_test_split(
    frame_sequences, angles, grade, test_size=0.2, random_state=42
)

# Input shapes
frame_input = Input(shape=(frame_sequences.shape[1],), name="frame_input")
angle_input = Input(shape=(1,), name="angle_input")

# Frame embedding
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 16
x1 = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(frame_input)
x1 = Flatten()(x1)

# Combine angle and frame features
x = Concatenate()([x1, angle_input])
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output = Dense(1)(x)

# Build and compile the model
model = Model(inputs=[frame_input, angle_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(
    [X_frames_train, X_angles_train],
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=([X_frames_test, X_angles_test], y_test)
)

# Save the model
model.save("difficulty_model.h5")

# Evaluate the model
loss, mae = model.evaluate([X_frames_test, X_angles_test], y_test, verbose=0)
print(f"Mean Absolute Error on test set: {mae:.4f}")

# Get predictions
y_pred = model.predict([X_frames_test, X_angles_test])

# Compute absolute errors and standard deviation
errors = np.abs(y_pred - y_test)
std_dev = np.std(errors)

print(f"Standard Deviation of Absolute Errors: {std_dev:.4f}")

# Compute absolute errors
errors = np.abs(y_pred - y_test)

# Plot the deviation chart
plt.figure(figsize=(12, 6))
plt.plot(errors, marker='o', linestyle='-', color='blue')
plt.title("Absolute Error (Deviation) for Each Test Sample")
plt.xlabel("Test Sample Index")
plt.ylabel("Absolute Error")
plt.grid(True)
plt.tight_layout()
plt.show()
