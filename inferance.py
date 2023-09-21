# inferance.py
from config import Config
from vectorizer import load_vectorizers
from model import load_model
from translate import translate

# Load the config, vectorizers, and model
config = Config()
eng_vectorizer, fra_vectorizer = load_vectorizers()
model = load_model(config.checkpoint_filepath)

# Example usage of the translate function
sentence = "Translate this English sentence to French."
translated_sentence = translate(sentence, config.sequence_length, config.target_vocab_size)
print("Translated sentence:", " ".join(translated_sentence))
