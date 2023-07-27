# lr_schedule.py
import tensorflow as tf

class LrSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate schedule with warmup for training.

    Args:
        hidden_size: Hidden size of the model.
        warmup_steps: Number of warmup steps for learning rate warmup.
    """

    def __init__(self, hidden_size, warmup_steps=4000):
        super(LrSchedule, self).__init__()
        self.warmup_steps = warmup_steps
        self.d = tf.cast(hidden_size, tf.float32)

    def __call__(self, step):
        """
        Calculates the learning rate based on the current step.

        Args:
            step: Current optimization step.

        Returns:
            The learning rate value.

        """
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        lr = tf.math.rsqrt(self.d) * tf.math.minimum(arg1, arg2)
        return lr

    def get_config(self):
        """
        Returns the configuration of the custom learning rate schedule.

        Returns:
            Configuration dictionary.

        """
        return {
            "warmup_steps": self.warmup_steps,
            "hidden_size": int(self.d.numpy()),  # Cast to native Python int
        }
