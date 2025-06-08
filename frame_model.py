import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import re
import json

def preprocess_sequence(input_string, max_items=20):
    """
    Preprocesses the input string containing multiple pXXXXrXX items.
    
    Args:
        input_string: String containing concatenated items (e.g., "p1100r15p1146r15...")
        max_items: Maximum number of items to consider (padding/truncating to this length)
    
    Returns:
        Numerical features array of shape (max_items, 2)
    """
    # Find all matches of the pattern pXXXXrXX
    items = re.findall(r'p(\d{4})r(\d{2})', input_string)
    
    # Convert to numerical values
    features = []
    for x_part, r_part in items[:max_items]:
        features.append([int(x_part)/10000.0, int(r_part)/100.0])
    
    # Pad with zeros if we have fewer than max_items
    while len(features) < max_items:
        features.append([0.0, 0.0])
    
    return np.array(features)

def create_sequence_model(max_items=20):
    """
    Creates a DNN model for processing sequences of items.
    
    Args:
        max_items: Maximum number of items in the sequence
    
    Returns:
        Compiled TensorFlow model
    """
    input_layer = layers.Input(shape=(max_items, 2))
    
    # Option 1: Flatten approach (simpler)
    x = layers.Flatten()(input_layer)
    x = layers.Dense(64, activation='relu')(x)
    
    # Option 2: LSTM approach (better for sequence relationships)
    # x = layers.LSTM(32, return_sequences=False)(input_layer)
    
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    output = layers.Dense(1)(x)
    
    model = models.Model(inputs=input_layer, outputs=output)
    
    model.compile(optimizer='adam',
                 loss='mse',
                 metrics=['mae'])
    
    return model

# Example usage with database data
if __name__ == "__main__":
    # Load data from the JSON file created by the database extraction script
    with open('training_data.json') as f:
        db_data = json.load(f)
    
    # Extract sequences and grades from database
    db_sequences = [item['frame'] for item in db_data]
    db_outputs = [float(item['grade']) for item in db_data]  # Ensure grades are floats
    
    # You can still keep sample data for testing if needed
    sample_sequences = [
        "p1100r15p1146r15p1154r15p1186r13",
        "p1100r15p1146r15p1154r15p1186r13p1191r12p1205r15p1215r15p1216r13p1241r12p1254r13p1280r13p1285r13p1332r13p1345r13p1379r14p1482r15p1519r15",
    ]
    sample_outputs = [12.5, 24]
    
    # Combine database data with sample data (optional)
    all_sequences = db_sequences + sample_sequences
    all_outputs = db_outputs + sample_outputs
    
    # Determine maximum sequence length in the data
    max_items = max(len(re.findall(r'p\d{4}r\d{2}', s)) for s in all_sequences)
    print(f"Maximum items in sequences: {max_items}")
    
    # Preprocess all sequences
    X = np.array([preprocess_sequence(s, max_items) for s in all_sequences])
    y = np.array(all_outputs)
    
    # Create and train model
    model = create_sequence_model(max_items)
    history = model.fit(X, y, 
                      epochs=100,  # Increased epochs for better learning
                      batch_size=32,  # Increased batch size
                      validation_split=0.2,
                      shuffle=True)
    
    # Save the trained model
    tf.saved_model.save(model, 'boulder_grade_predictor')  # SavedModel format
    print("Model saved as boulder_grade_predictor.h5")
    
    # Test prediction with a sequence from the database
    if len(db_sequences) > 0:
        test_sequence = db_sequences[0]  # Use first database sequence for testing
        test_input = np.array([preprocess_sequence(test_sequence, max_items)])
        prediction = model.predict(test_input)
        actual_grade = db_outputs[0]
        print(f"\nTest Prediction for sequence (first db entry):")
        print(f"Input: {test_sequence}")
        print(f"Predicted grade: {prediction[0][0]:.2f}")
        print(f"Actual grade: {actual_grade:.2f}")
        print(f"Difference: {abs(prediction[0][0] - actual_grade):.2f}")
    
    # You can also test with your sample sequences
    test_sequence = "p1100r15p1146r15p1154r15p1186r13p1191r12p1205r15p1215r15p1216r13p1241r12p1254r13p1280r13p1285r13p1332r13p1345r13p1379r14p1482r15p1519r15"
    test_input = np.array([preprocess_sequence(test_sequence, max_items)])
    prediction = model.predict(test_input)
    print(f"\nPrediction for sample sequence {test_sequence}: {prediction[0][0]:.2f}")