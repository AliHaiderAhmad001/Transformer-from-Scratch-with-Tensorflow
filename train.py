# train.py
import tensorflow as tf
from config import Config
from transformer import Transformer
from lr_schedule import LrSchedule
from transformer_callbacks import TransformerCallbacks
from loss_functions import cce_loss
from metrics import masked_accuracy
from data_preprocessing import prepare_dataset, format_dataset, make_dataset
from vectorizer import create_vectorizers, save_vectorizers

text_file = "dataset/fra.txt"

config = Config()

# Prepare the dataset
text_pairs = prepare_dataset(text_file)
max_len = max([max([len(x[0].split()), len(x[1].split())]) for x in text_pairs])
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
random.shuffle(text_pairs)
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

# Create the vectorizers
eng_vectorizer, fra_vectorizer = create_vectorizers(config)

# Learn the vocabulary
train_eng_texts = [pair[0] for pair in train_pairs]
train_fra_texts = [pair[1] for pair in train_pairs]
eng_vectorizer.adapt(train_eng_texts)
fra_vectorizer.adapt(train_fra_texts)

# Save the vectorizers
save_vectorizers(eng_vectorizer, fra_vectorizer)

# Make datasets
train_ds = make_dataset(train_pairs, eng_vectorizer, fra_vectorizer)
val_ds = make_dataset(val_pairs, eng_vectorizer, fra_vectorizer)
test_ds = make_dataset(test_pairs, eng_vectorizer, fra_vectorizer)


# Create the Transformer model
transformer = Transformer(config, config.source_vocab_size, config.target_vocab_size)

# Create the learning rate schedule
lr = LrSchedule(config.hidden_size, config.warmup_steps)

# Create the custom callbacks for monitoring and early stopping
callbacks = TransformerCallbacks(config)

# Create the Adam optimizer with the custom learning rate
optimizer = tf.keras.optimizers.Adam(
    lr,
    beta_1=0.9,
    beta_2=0.98,
    epsilon=1e-9
)

# Compile the Transformer model with the custom loss function and optimizer
transformer.compile(
    loss=cce_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy]
)

# Train the Transformer model on the training dataset
# and validate it on the validation dataset
history = transformer.fit(
    train_ds,
    epochs=config.epochs,
    validation_data=val_ds,
    callbacks=callbacks
)
