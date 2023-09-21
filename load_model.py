# model.py

import tensorflow as tf
from config import Config
from encoder import Encoder
from decoder import Decoder
from embeddings import Embeddings
from lr_schedule import LrSchedule
from transformer import Transformer
from feed_forward import FeedForward
from attention import AttentionHead, MultiHead_Attention
from positional_embeddings import PositionalEmbeddings, SinusoidalPositionalEncoding

def load_model(model_path):
    """
    Load the Transformer model from the given model path.

    Args:
        model_path (str): The path to the saved Transformer model.

    Returns:
        tf.keras.Model: The loaded Transformer model.
    """
    config = Config()
    custom_objects = {
        "LrSchedule": LrSchedule,
        "PositionalEmbeddings": PositionalEmbeddings,
        "Embeddings": Embeddings,
        "AttentionHead": AttentionHead,
        "MultiHead_Attention": MultiHead_Attention,
        "FeedForward": FeedForward,
        "Encoder": Encoder,
        "Decoder": Decoder,
        "Transformer": Transformer,
        "cce_loss": cce_loss,
        "masked_accuracy": masked_accuracy}

    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(model_path)

    return model
