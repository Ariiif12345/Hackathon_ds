import pickle
import numpy as np
import os

# Load encoders.pkl (your file has dicts inside)
encoder_path = os.path.join(os.path.dirname(__file__), 'encoders.pkl')

with open(encoder_path, 'rb') as f:
    encoders = pickle.load(f)

def get_reverse_mapping(mapping):
    """Helper function to reverse a dictionary."""
    return {v: k for k, v in mapping.items()}

def encode_inputs(enrollment, duration, phase, sponsor_type, gender, condition, location):
    """
    Encode raw inputs using saved mapping dicts.
    Handles both numeric and string input for phase.
    """

    # Handle numeric phase input (e.g., 2 -> 'Phase 3')
    phase_mapping = encoders['phase']
    if isinstance(phase, int):
        phase_reverse = get_reverse_mapping(phase_mapping)
        if phase not in phase_reverse:
            raise ValueError(f"Invalid numeric phase input: {phase}")
        phase = phase_reverse[phase]

    try:
        phase_encoded = phase_mapping[phase]
        sponsor_encoded = encoders['sponsor_type'][sponsor_type]
        gender_encoded = encoders['gender'][gender]
        condition_encoded = encoders['condition'][condition]
        location_encoded = encoders['location'][location]
    except KeyError as e:
        raise ValueError(f"Invalid input for encoding: {e}")

    features = np.array([
        enrollment,
        duration,
        phase_encoded,
        sponsor_encoded,
        gender_encoded,
        condition_encoded,
        location_encoded
    ]).reshape(1, -1)

    return features
