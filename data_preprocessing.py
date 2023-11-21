# data_preprocessing.py
import random
import re
import unicodedata
import pickle
from config import Config
import tensorflow as tf
from vectorizer import load_vectorizers

config = Config()

def normalize(line):
    """
    Normalize a line of text and split into two at the tab character
    Args: The normalize function takes a line of text as input.
    Return: Normalized English and French sentences as a tuple (eng, fra).
    """
    line = unicodedata.normalize("NFKC", line.strip())

    # Perform regular expression substitutions to add spaces around non-alphanumeric characters
    line = re.sub(r"^([^ \w])(?!\s)", r"\1 ", line)
    line = re.sub(r"(\s[^ \w])(?!\s)", r"\1 ", line)
    line = re.sub(r"(?!\s)([^ \w])$", r" \1", line)
    line = re.sub(r"(?!\s)([^ \w]\s)", r" \1", line)

    # Split the line of text into two parts at the tab character
    x = line.split("\t")
    eng, fra = x[0], x[1]
    # Add "[start]" and "[end]" tokens to the "fra" part of the line
    fra = "[start] " + fra + " [end]"

    # Return the normalized English and French sentences
    return eng, fra


def prepare_dataset(text_file):
    """
    Reads the text file, normalizes and splits the lines, and prepares the dataset.

    Args:
        text_file (str): The path to the text file containing English and French sentences.

    Returns:
        list: A list of tuples containing normalized English and French sentences.
    """
    with open(text_file) as fp:
        lines = fp.readlines()

    text_pairs = [normalize(line) for line in lines]
    return text_pairs


def make_dataset(pairs, batch_size=64):
    """
    Creates a dataset from pairs of English and French texts.

    Args:
        pairs: List of pairs containing English and French texts.
        batch_size: Batch size for the dataset.

    Returns:
        Formatted and preprocessed dataset.
    """
    eng_vectorizer, fra_vectorizer = load_vectorizers(config.vectorizers_path)

    def format_dataset(eng, fra):
        """
        Formats the dataset by applying vectorization and preparing the inputs for the encoder and decoder.

        Args:
            eng: English text tensor.
            fra: French text tensor.

        Returns:
            Tuple of formatted inputs for the encoder and decoder.
        """
        eng = eng_vectorizer(eng)
        fra = fra_vectorizer(fra)
        return (
            {"encoder_inputs": eng, "decoder_inputs": fra[:, :-1]},
            fra[:, 1:]
        )
    eng_texts, fra_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    fra_texts = list(fra_texts)

    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, fra_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(2048).prefetch(tf.data.experimental.AUTOTUNE).cache()

    return dataset
