# config.py

class Config:
    def __init__(self):
        """
        Configuration class for transformer model hyperparameters.

        Attributes:
            sequence_length: Maximum sequence length for input sequences.
            hidden_size: Hidden size for the transformer model.
            frequency_factor: Frequency factor for positional encodings.
            source_vocab_size: Vocabulary size for the source language.
            target_vocab_size: Vocabulary size for the target language.
            positional_information_type: Type of positional embeddings to use.
            hidden_dropout_prob: Dropout probability for the hidden layers.
            num_heads: Number of attention heads in multi-head attention.
            intermediate_fc_size: Size of the intermediate fully connected layer.
            warmup_steps: Number of warm-up steps for learning rate scheduling.
            num_blocks: Number of encoder and decoder blocks in the transformer.
            final_dropout_prob: Dropout probability for the final output.
            epochs: Number of epochs for training.
            patience: Number of epochs with no improvement after which training will be stopped.
            checkpoint_filepath: Filepath for saving model checkpoints.
            vectorizers_path: Filepath for saving vectorizers.
            dataset_path: Filepath for dataset.
        """
        self.sequence_length = 60
        self.hidden_size = 256
        self.frequency_factor = 10000
        self.source_vocab_size = 16721
        self.target_vocab_size = 31405
        self.positional_information_type = 'embs'
        self.hidden_dropout_prob = 0.1
        self.num_heads = 4
        self.intermediate_fc_size = self.hidden_size * 4
        self.warmup_steps = 4000
        self.num_blocks = 2
        self.final_dropout_prob = 0.5
        self.epochs = 30
        self.patience = 3
        self.checkpoint_filepath = 'tmp/checkpoint'
        self.vectorizers_path = 'tmp/vectorize.pickle'
        self.dataset_path = "dataset/fra.txt"
        
