def generate_humming_embedding():
    pass

def generate_classical_embedding():
    pass

humming_embedding = generate_humming_embedding()
classical_embedding = generate_classical_embedding()

def humming_to_classical_embedding(audio_embedding):
    return audio_embedding - humming_embedding + classical_embedding

def classical_to_humming_embedding(audio):
    return audio_embedding - classical_embedding + humming_embedding