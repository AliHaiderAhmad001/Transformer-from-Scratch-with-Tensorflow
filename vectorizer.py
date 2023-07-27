# vectorizer.py
import pickle
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

def create_vectorizers(config):
    """
    Create and configure the vectorizers for English and French languages.

    Args:
        config: Configuration object containing model hyperparameters.

    Returns:
        eng_vectorizer: TextVectorization for English language.
        fra_vectorizer: TextVectorization for French language.
    """
    eng_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=config.source_vocab_size,
        standardize=None,
        split="whitespace",
        output_mode="int",
        output_sequence_length=config.sequence_length,
    )

    fra_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=config.target_vocab_size,
        standardize=None,
        split="whitespace",
        output_mode="int",
        output_sequence_length=config.sequence_length + 1  # since we'll need to offset the sentence by one step during training.
    )

    return eng_vectorizer, fra_vectorizer


def save_vectorizers(eng_vectorizer, fra_vectorizer, save_path="tmp/vectorize.pickle"):
    """
    Save the English and French vectorizers to a pickle file.

    Args:
        eng_vectorizer: The English TextVectorization instance.
        fra_vectorizer: The French TextVectorization instance.
        save_path: The file path to save the vectorizers.
    """
    data = {
        "engvec_config": eng_vectorizer.get_config(),
        "engvec_weights": eng_vectorizer.get_weights(),
        "fravec_config": fra_vectorizer.get_config(),
        "fravec_weights": fra_vectorizer.get_weights(),
    }

    with open(save_path, "wb") as fp:
        pickle.dump(data, fp)

def load_vectorizers(file_path="tmp/vectorize.pickle"):
    with open(file_path, "rb") as fp:
        data = pickle.load(fp)

    eng_vectorizer = tf.keras.layers.TextVectorization.from_config(data["engvec_config"])
    eng_vectorizer.set_weights(data["engvec_weights"])
    fra_vectorizer = tf.keras.layers.TextVectorization.from_config(data["fravec_config"])
    fra_vectorizer.set_weights(data["fravec_weights"])

    return eng_vectorizer, fra_vectorizer
