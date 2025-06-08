import tensorflow as tf
import numpy as np
import re

def preprocess_sequence(input_string, max_items):
    """The same preprocessing function used during training"""
    items = re.findall(r'p(\d{4})r(\d{2})', input_string)
    features = []
    for x_part, r_part in items[:max_items]:
        features.append([int(x_part)/10000.0, int(r_part)/100.0])
    while len(features) < max_items:
        features.append([0.0, 0.0])
    return np.array(features)

# Load the saved model
model = tf.keras.models.load_model('boulder_grade_predictor.h5')

# Sample cases to predict
sample_problems = [
    "p1100r15p1146r15p1154r15p1186r13p1191r12p1205r15p1215r15p1216r13p1241r12p1254r13p1280r13p1285r13p1332r13p1345r13p1379r14p1482r15p1519r15"  
]


# Use the same max_items as during training (should match what you used)
max_items = 10  # Adjust this to match your training setting

for problem in sample_problems:
    # Preprocess the problem
    processed = preprocess_sequence(problem, max_items)
    input_data = np.array([processed])  # Add batch dimension
    
    # Make prediction
    predicted_grade = model.predict(input_data)[0][0]
    
    print(f"Problem: {problem}")
    print(f"Predicted grade: {predicted_grade:.2f}\n")