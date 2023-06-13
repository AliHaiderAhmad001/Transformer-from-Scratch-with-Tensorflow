# Neural-Machine-Translator with Transformers
The task of translating text from one language into another is a common task. In this repo I'm going to construct and demonstrate a naive model for an English to French translation task using TensorFlow, which is inspired by the model presented in the "Attention Is All You Need" paper.

تعتبر مهمة ترجمة النص من لغة إلى لغة أخرى من المهام الشائعة. سأقوم في هذا الريبو ببناء وشرح نموذج بسيط لمهمة الترجمة من اللغة الإنجليزية إلى اللغة  الفرنسية باستخدام تنسرفلو، وهو مستلهم من النموذج الذي تم تقديمه في ورقة "الانتباه هو كل ماتحتاجه".

## Transformer architecture
The Transformer architecture is a popular model architecture used for various natural language processing tasks, including machine translation, text generation, and language understanding. It was introduced by Vaswani et al. in the paper "Attention Is All You Need."

The Transformer architecture consists of two main components: the encoder and the decoder. Both the encoder and decoder are composed of multiple layers of self-attention and feed-forward neural networks. The key idea behind the Transformer is the use of self-attention mechanisms, which allow the model to focus on different parts of the input sequence when generating the output:

المحولات هي بنية نموذجية شائعة تستخدم في العديد من مهام معالجة اللغة الطبيعية، بما في ذلك الترجمة الآلية وتوليد النص وفهم اللغة. تم تقديمه بواسطة Vaswani et al. في مقالة "الاهتمام هو كل ما تحتاجه".

تتكون بنية المحول من مكونين رئيسيين: المشفر ووحدة فك التشفير. يتكون كل من المشفر ومفكك التشفير من طبقات متعددة من الاهتمام الذاتي والشبكات العصبية ذات التغذية الأمامية. الفكرة الرئيسية وراء المحول هي استخدام آليات الانتباه الذاتي، والتي تسمح للنموذج بالتركيز على أجزاء مختلفة من سلسلة الدخل عند إنشاء سلسلة الخرج:


![Transformer architecture](imgs/transformer_arch.png "Transformer architecture")


**Here is a high-level overview of the Transformer architecture:**

**1. Encoder:**
* The input sequence is first passed through an `embedding layer`, which maps each token to a continuous vector representation -followed by `positional encoding` to provide information about the position of tokens in the sequence.
* The embedded input is then processed by a stack of identical `encoder layers`. Each `encoder layer` consists of a `multi-head self-attention` mechanism, followed by a `feed-forward` neural network.
* The `self-attention` mechanism allows the `encoder` to attend to different positions in the input sequence and capture dependencies between tokens.
* The output of the `encoder` is a set of encoded representations for each token in the input sequence.

فيما يلي نظرة عامة عالية المستوى على بنية المحولات:

تُمرّر سلسلة الدخل إلى `طبقة تضمين` تقوم بربط كل وحدة نصية (كلمات في حالتنا) بشعاع رقمي يُمثّلها مكون من قيم مستمرة (أعداد حقيقية). يتبع ذلك `ترميّز موضعي` يهدف إلى تقديم معلومات حول مواقع الوحدات النصية ضمن سلسلة الدخل. تجري معالجة هذه التضمينات بعد ذلك من خلال `طبقات المشفّر` المتماثلة. كل `طبقة تشفير` تتألف من طبقة `انتباه ذاتي متعدد الرؤوس` متبوعة بطبقة `تغذية أمامية`.

تسمح آلية الانتباه الذاتي `للمشفّر` بمشاهدة المواقع المختلفة لسلسلة الدخل، والتقاط التبعيات بين الوحدات النصية. أي بمعنى آحر، تسمح له بمعرفة مدى ارتباط كل كلمة مع الكلمات الأخرى. خرج `المشفّر` هو تمثيّل جديد لكل وحدة نصيّة من سلسلة الدخل.

**2. Decoder:**
* The `decoder` takes the encoded representations from the `encoder` as input, along with the target sentence.
* Similar to the `encoder`, the input to the `decoder` is first passed through an `embedding layer`, followed by `positional encoding`.
* The `decoder` also consists of a stack of identical `decoder layers`. Each `decoder layer` has two self-attention mechanisms: a `masked multi-head self-attention` mechanism to prevent the `decoder` from looking ahead, and a `cross-attention` mechanism that allows the `decoder` to attend to the encoder's output.
* The `decoder` generates the output sequence.

تأخذ وحدة `فك التشفير` التمثيلات من `المشفّر` كدخل، جنبًا إلى جنب مع الجملة الهدف. كما في وحدة `المُشفّر`؛ تُمرّر سلسلة الدخل إلى `طبقة تضمين` تقوم بربط كل وحدة نصية (كلمات في حالتنا) بشعاع رقمي يُمثّلها مكون من قيم مستمرة (أعداد حقيقية). يتبع ذلك `ترميّز موضعي`.

تحتوي كل طبقة من طبقات وحدة `فك التشفير` على آليتين للانتباه الذاتي: آلية `الانتباه الذاتي متعدد الرؤوس المُقنّعة` لمنع وحدة `فك التشفير` من النظر إلى الأمام، وآلية `الانتباه المتبادل` التي تسمح ل`وحدة فك التشفير` بالوصول إلى خرج `المشفّر`. تقوم وحدة `فك التشفير` بتوليد سلسلة الخرج (الجملة بعد الترجمة).


**3. Positional Encoding:**
* `Positional encoding` is used to inject information about the position of tokens into the model.
* The `positional encoding` is added to the embedded input and provides the model with the positional information necessary for capturing sequential relationships.


يُستخدم `الترميز الموضعي` لإضافة معلومات المواقع للوحدات النصية إلى النموذج (بدونها لايمكن للنموذح فهم ترتيب الكلمات في الجملة).

**4. Final Linear Layer:**
* The output of the `decoder` is passed through a linear layer followed by a `Softmax` activation function to produce the probability distribution over the target vocabulary.

يتم تمرير خرج وحدة `فك التشفير` إلى طبقة خطية متبوعة بدالة تنشيط `Softmax` لإنتاج التوزيع الاحتمالي على المفردات المستهدفة.


The following figure shows encoder-decoder architecture of the transformer, with the encoder shown in the upper half of the figure and the decoder in the lower half:

**Note:** This figure represents the the "inference phase". Here it should be noted that the decoder’s mechanism in the inference phase is slightly different in the training phase, and we will understand that later.

يوضّح الشكل التالي بنية المُشفّر-فك التشفير في المحوّل، حيث يبيّن النصف العلوي من الشكل بنية المُشفّر، ووحدة فك التشفير في النصف السفلي: 

** ملاحظة: ** هذا الشكل يمثل "مرحلة الاستدلال". هنا تجدر الإشارة إلى أن آلية فك التشفير في مرحلة الاستدلال تختلف قليلاً في مرحلة التدريب، وسنفهم ذلك لاحقًا.


![Encoder-decoder examble](imgs/Transformer.png "Encoder-decoder examble")


**We’ll look at each of the components in detail shortly, but we can already see a few things in Figure above that characterize the Transformer architecture:**
* The input text is tokenized and converted to token embeddings. Since the attention mechanism is not aware of the relative positions of the tokens, we need a way to inject some information about token positions into the input to model the sequential nature of text. The token embeddings are thus combined with positional embeddings that contain positional information for each token.
* The encoder is composed of a stack of encoder layers or "blocks", which is analogous to stacking convolutional layers in computer vision. The same is true of the decoder, which has its own stack of decoder layers.
* The encoder's output is fed to each decoder layer, and the decoder then generates a prediction for the most probable next token in the sequence. The output of this step is then fed back into the decoder to generate the next token, and so on until a special end-of-sequence (EOS) token is reached. In the example from Figure above, imagine the decoder has already predicted "Die" and "Zeit". Now it gets these two as an input as well as all the encoder's outputs to predict the next token, "fliegt". In the next step the decoder gets "fliegt" as an additional input. We repeat the process until the decoder predicts the EOS token or we reached a maximum length.

يتم تقطيع النص المُدخل إلى وحدات نصية (يمكن اعتبارها كلمات) ثم يتم ربط كل وحدة نصية بشعاع تضمين يمثّلها. بما أن آلية الانتباه لاتستطيع التعرّف على مواقع الوحدات النصية، فإننا بحاجة إلى طريقةٍ ماتمكننا من إضافة بعض المعلومات التي تعبر عن مواقع هذه الوحدات إلى تمثيلات الوحدات النصية السابقة. يتم ذلك من خلال طبقة الترميز الموضعي، حيث يتم ربط كل موقع ضمن التسلسل بشعاع يُضاف إلى تمثيل الوحدة النصية الموافقة لذلك الموقع.

يتألف المُشفّر من عدة طبقات متماثلة من حيث البنية ومكدسة فوق بعضها بطريقة مشابهة لتكديس طبقات CNN، كل منها تدعى "طبقة تشفير" أو "كِتل". يتم تمرير تمثيلات الوحدات النصية إلى هذه الكتل، ويكون خرجها تمثيل جديد أكثر قوة للوحدات النصية.

يتم تمرير خرج وحدة المُشفّر (تمثيلات الوحدات النصية لجملة الدخل) إلى كل طبقة فك تشفير في وحدة فك التشفير (طبقة فك التشفير تتألف من عدة طبقات فك تشفير، أي بشكل مشابه لوحدة التشفير). تقوم وحدة فك التشفير بتوليد توقع يُمثّل الوحدة النصية التالية في جملة الهدف -الأكثر رجوحًا. تستمر عملية التوليد هذه وصولًا إلى رمز نهاية السلسلة EOS. في المثال الموضّح في الشكل أعلاه، تخيل أن وحدة فك التشفير توقعت كلمة "Die" وكلمة "Zeit". الآن سيتم أخذ هذه الكلمات كدخل إلى وحدة فك التشفير جنبًا إلى جنب مع خرج المُشفّر لتوقع الوحدة النصية التالية والتي هي "fliegt". في الخطوة التالية سيتم استخدام الكلمات الجديدة جنبًا إلى جنبًا مع الكلمات السابقة وخرج المشفر لتوليد الكلمة التالية. يتم تكؤار هذه العملية حتى يتم توقع رمز نهاية الجملة EOS.


## Model Implementation
Here are the steps to build an English to French translation model using the Transformers architecture

### Download dataset

We'll be working with an [English-to-French translation dataset](https://ankiweb.net/shared/decks/french):
```
import tensorflow as tf, pathlib
text_file = tf.keras.utils.get_file(
    fname="fra-eng.zip",
    origin="http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip",
    extract=True,
)
# File location
text_file = pathlib.Path(text_file).parent / "fra.txt"
```
The dataset we're working on consists of 167,130 lines. Each line consists of the original sequence (the sentence in English) and the target sequence (in French).

### Data preparation
We prepend the token "[start]" and we append the token "[end]" to the French sentence.

```
import pathlib
import pickle
import random
import re
import unicodedata

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
    eng, fra = line.split("\t")
    
    # Add "[start]" and "[end]" tokens to the "fra" part of the line
    fra = "[start] " + fra + " [end]"
    
    # Return the normalized English and French sentences
    return eng, fra


# normalize each line and separate into English and French
with open(text_file) as fp:
    text_pairs = [normalize(line) for line in fp]
with open("text_pairs.pickle", "wb") as fp:
    pickle.dump(text_pairs, fp)
```
**Explanation of the code:**
1. The `normalize` function takes a line of text as input.
2. It first applies Unicode normalization form NFKC to the line, which converts characters to their standardized forms
   (e.g., converting full-width characters to half-width).
3. The regular expression substitutions using `re.sub` are performed to add spaces around non-alphanumeric characters in the line.
   The patterns and replacement expressions are as follows.
4. The line is then split into two parts at the tab character using line.split("\t"), 
   resulting in the English sentence (eng) and the French sentence (fra).
5. The [start] and [end] tokens are added to the fra part of the line to indicate the start and end of the sentence.

**Let's look at the data:**
```
print(f"Each sample of data will look like this: {text_pairs[55805]}")
print(f"Numper of sampels: {len(text_pairs)}")
print(f"Max length in english sequences: {max([len(x[0].split()) for x in text_pairs])}")
print(f"Max length in french sequences: {max([len(x[1].split()) for x in text_pairs])}")
print(f"Numper of token in english sequences: {len(set(token for s in [x[0].split() for x in text_pairs] for token in s))}")
print(f"Numper of token in french sequences: {len(set(token for s in [x[1].split() for x in text_pairs] for token in s))}")
```
```
Output:
Each sample of data will look like this: ('what does he see in her ?', '[start] que lui trouve-t-il  ?  [end]')
Numper of sampels: 167130
Max length in english sequences: 51
Max length in french sequences: 60
Numper of token in english sequences: 14969
Numper of token in french sequences: 29219
```

### Vectorizing the text data

We need to write a function that associates each token with a unique integer number representing it to get what is called a "Tokens_IDs". Fortunately, there is a layer in TensorFlow called [`TextVectorization`](https://keras.io/api/layers/preprocessing_layers/core_preprocessing_layers/text_vectorization/) that makes life easier for us. We'll use two instances of the TextVectorization layer to vectorize the text data (one for English and one for Spanish). 

**First of all, let's split the sentence pairs into a training set, a validation set, and a test set:**

```
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")
```
```
Output:
167130 total pairs
116992 training pairs
25069 validation pairs
25069 test pairs
```

**Now we'll do Vectorization:**

```
from tensorflow.keras.layers import TextVectorization

vocab_size_en = 14969
vocab_size_fr = 29219
seq_length = 60

# English layer
eng_vectorizer = TextVectorization(
    max_tokens=vocab_size_en,
    standardize=None,
    split="whitespace",
    output_mode="int",
    output_sequence_length=seq_length,
)
# French layer
fra_vectorizer = TextVectorization(
    max_tokens=vocab_size_fr,
    standardize=None,
    split="whitespace",
    output_mode="int",
    output_sequence_length=seq_length + 1 # since we'll need to offset the sentence by one step during training.
                                            
)

train_eng_texts = [pair[0] for pair in train_pairs]
train_fra_texts = [pair[1] for pair in train_pairs]
# Learn the vocabulary
eng_vectorizer.adapt(train_eng_texts)
fra_vectorizer.adapt(train_fra_texts)
```

### Making dataset

**Now we have to define how we will pass the data to the model. There are several ways to do this:**

* Present the data as a NumPy array or a tensor (Faster, but need to load all data into memory).
* Create a Python generator function and let the loop read data from it (Fetched from the hard disk when needed, rather than being loaded all into memory).
* Use the `tf.data` dataset.

**We'll choose the fourth manner. The general benefits of using the `tf.data` dataset are:**

* The flexibility in handling the data.
* It makes feeding the model with data more efficient and fast.

#### What is `tf.data`?

I'm gonna be brief..

`tf.data` is a module in TensorFlow that provides tools for building efficient and scalable input pipelines for machine learning models. It is designed to handle large datasets, facilitate data preprocessing, and enable high-performance data ingestion for training and evaluation. Using tf.data, you can build efficient and scalable input pipelines for training deep learning models.

**Here are some important functions:**

* `shuffle(n)`: Randomly fills a buffer of data with `n` data points and randomly shuffles the data in the buffer. When data is pulled out of the buffer (such as when grabbing the next batch of data), TensorFlow automatically refills the buffer.
* `batch(n)`: Generate batches of the dataset, each of size n.
* `prefetch(n)`: to keep n batches/elements in memory ready for the training loop to consume.
* `cache(): Efficiently caches the dataset for faster subsequent reads.
* `map(func)`: Applying a transform (function) on data batches.
* [You can read more here](https://pyimagesearch.com/2021/06/14/a-gentle-introduction-to-tf-data-with-tensorflow/).
* [ِAnd here](https://stackoverflow.com/questions/76414594/shuffle-the-batches-in-tensorflow-dataset/76443517#76443517).
    
```
from tensorflow.data import AUTOTUNE

def format_dataset(eng, fra):
    eng = eng_vectorizer(eng)
    fra = fra_vectorizer(fra)
    return ({"encoder_inputs": eng, "decoder_inputs": fra[:, :-1],},
            fra[:, 1:])
def make_dataset(pairs, batch_size=64):
    eng_texts, fra_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    fra_texts = list(fra_texts)
    
    # Convert the lists to TensorFlow tensors
    eng_texts = tf.convert_to_tensor(eng_texts)
    fra_texts = tf.convert_to_tensor(fra_texts)
    
    # Create a TensorFlow dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, fra_texts))
    
    # Apply dataset transformations
    dataset = dataset.shuffle(len(pairs))  # Shuffle the entire dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(AUTOTUNE).cache()
    
    return dataset

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)
```

**Let's take a look:**
```
for inputs, targets in train_ds.take(1):
    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    print(f'inputs["encoder_inputs"][0]: {inputs["encoder_inputs"][0]}')
    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    print(f'inputs["decoder_inputs"][0]: {inputs["decoder_inputs"][0]}')
    print(f"targets.shape: {targets.shape}")
    print(f"targets[0]: {targets[0]}")
```
```
Output:
inputs["encoder_inputs"].shape: (64, 60)
inputs["encoder_inputs"][0]: [   3  305  862 1192  559    7  167  182    2    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0]
inputs["decoder_inputs"].shape: (64, 60)
inputs["decoder_inputs"][0]: [  2   6  82   8 436  13 821 527 172   4   3   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0]
targets.shape: (64, 60)
targets[0]: [  6  82   8 436  13 821 527 172   4   3   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0]
```

Now, we have our data ready to be fed into a model.

### Positional information

    Positional encoding is a technique used in transformers to incorporate the positional information of words or tokens into the input embeddings. Since transformers don't have any inherent notion of word order, positional encoding helps the model understand the sequential order of the input sequence. In transformers, positional encoding is typically added to the input embeddings before feeding them into the encoder or decoder layers. The positional encoding vector is added element-wise to the token embeddings, providing each token with a unique position-dependent representation. There are three common types of positional encodings used in transformers:

1. **Learned Positional Embeddings:** Instead of using fixed sinusoidal functions, learned positional embeddings are trainable parameters that are optimized during the model training. These embeddings can capture position-specific patterns and dependencies in the input sequence.

2. **Relative Positional Encodings:** In addition to absolute positional information, relative positional encodings take into account the relative distances or relationships between tokens. They can be used to model the dependencies between tokens at different positions in a more explicit manner.

3. **Hybrid Approaches:** Some models combine both learned positional embeddings and sinusoidal positional encodings to capture both the learned and fixed positional information. This hybrid approach can provide a flexible representation while also allowing the model to benefit from the sinusoidal encoding's ability to generalize across sequence lengths.

The choice of positional encoding depends on the specific task, dataset, and model architecture. Sinusoidal positional encoding is widely used and has shown good performance in various transformer-based models. However, experimenting with different types of positional encodings can be beneficial to improve the model's ability to capture positional information effectively.

In this section we will investigate the two approaches: **Learned Positional Embeddings** and ٍ **Sinusoidal positional encoding**. However, it may not be necessary to use complex positional encoding methods. The standard sinusoidal positional encoding used in the original Transformer model can still work well.

#### Sinusoidal positional encoding

    The most commonly used method for positional encoding in transformers is the sinusoidal positional encoding, as introduced in the "Attention Is All You Need" paper by Vaswani et al. The sinusoidal positional encoding is based on the idea that different positions can be represented by a combination of sine and cosine functions with different frequencies. The formula for the sinusoidal positional encoding is as follows:

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

where PE(pos, 2i) represents the i-th dimension of the positional encoding for the token at position "pos", and d_model is the dimensionality of the model.

```
import tensorflow as tf

def create_positional_encoding_matrix(sequence_length, embedding_dimension, frequency_factor=10000):
    """
    Create a positional encoding matrix.

    Args:
        sequence_length (int): Length of the input sequence.
        embedding_dimension (int): Dimensionality of the positional embeddings. Must be an even integer.
        frequency_factor (int): Constant for the sinusoidal functions.

    Returns:
        tf.Tensor: Matrix of positional embeddings of shape (sequence_length, embedding_dimension).
        The value at element (k, 2i) is sin(k / frequency_factor^(2i / embedding_dimension)),
        and the value at element (k, 2i+1) is cos(k / frequency_factor^(2i / embedding_dimension)).
    """
    assert embedding_dimension % 2 == 0, "Embedding dimension needs to be an even integer"
    embedding_dimension_half = embedding_dimension // 2
    positions = tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis]  # Column vector of shape (sequence_length, 1)
    frequency_indices = tf.range(embedding_dimension_half, dtype=tf.float32)[tf.newaxis, :]  # Row vector of shape (1, embedding_dimension/2)
    frequency_denominator = tf.pow(frequency_factor, -frequency_indices / embedding_dimension_half)  # frequency_factor^(-2i/d)
    frequency_arguments = positions / frequency_denominator  # Matrix of shape (sequence_length, embedding_dimension)
    sin_values = tf.sin(frequency_arguments)
    cos_values = tf.cos(frequency_arguments)
    positional_encodings = tf.concat([sin_values, cos_values], axis=1)
    return positional_encodings
    
class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    """
    SinusoidalPositionalEncoding layer.

    This layer applies sinusoidal positional encodings to the input embeddings.

    Args:
        config (object): Configuration object containing parameters.
    """
    def __init__(self, config, **kwargs):
        """
        Initialize the SinusoidalPositionalEncoding layer.

        Args:
            config (object): Configuration object with parameters for positional encoding.
        """
        super().__init__(**kwargs)
        self.sequence_length = config.sequence_length
        self.position_encoding = create_positional_encoding_matrix(
            config.sequence_length, config.hidden_size, config.frequency_factor
        )

    def call(self, input_ids):
        """
        Apply positional encodings to the input embeddings.

        Args:
            input_ids (tf.Tensor): Input tensor containing token IDs.

        Returns:
            tf.Tensor: Output tensor with positional encodings added.
        """
        if input_ids.shape[1] != self.sequence_length:
            self.position_encoding = create_positional_encoding_matrix(
                input_ids.shape[1], config.hidden_size, config.frequency_factor
            )
        return self.position_encoding 

    def compute_mask(self, input_ids, mask=None):
        """
        Compute the mask for the input IDs.

        Args:
            input_ids (tf.Tensor): Input tensor containing token IDs.
            mask (tf.Tensor): Optional mask tensor.

        Returns:
            tf.Tensor: Computed mask tensor.
        """
        if not self.mask_zero:
            return None
        return tf.not_equal(input_ids, 0)

    def get_config(self):
        """
        Get the configuration of the SinusoidalPositionalEncoding layer.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "position_embeddings": self.position_embeddings,
        })
        return config
```

**Testing:**
```
# Define the configuration
class Config:
    def __init__(self):
        self.sequence_length = 4
        self.hidden_size = 4
        self.frequency_factor = 10000
        self.mask_zero = True

config = Config()

# Create an instance of the SinusoidalPositionalEncoding layer
positional_encoding_layer = SinusoidalPositionalEncoding(config)

# Create a sample input tensor with token IDs
batch_size = 1
seq_length = 4
input_ids = tf.random.uniform((batch_size, seq_length), maxval=config.sequence_length, dtype=tf.int32)

# Apply positional encodings
output_embeddings = positional_encoding_layer(input_ids)

# Print the output positional embeddings
print("Outputs:")
print(output_embeddings)
print(input_ids)
print(output_embeddings._keras_mask)
"""
Outputs:
tf.Tensor(
[[ 0.          0.          1.          1.        ]
 [ 0.84147096 -0.50636566  0.5403023   0.8623189 ]
 [ 0.9092974  -0.87329733 -0.4161468   0.48718765]
 [ 0.14112    -0.99975586 -0.9899925  -0.02209662]], shape=(4, 4), dtype=float32)
tf.Tensor([[2 1 0 3]], shape=(1, 4), dtype=int32)
tf.Tensor([[ True  True False  True]], shape=(1, 4), dtype=bool)
"""
```

By using sinusoidal positional encoding, the model can differentiate between tokens based on their positions in the input sequence. This allows the transformer to capture sequential information and attend to different parts of the sequence appropriately. It's important to note that positional encoding is added as a fixed representation and is not learned during the training process. The model learns to incorporate the positional information through the attention mechanism and the subsequent layers of the transformer.

#### Learned Positional Embeddings

    Learned positional embeddings refer to the practice of using trainable parameters to represent positional information in a sequence. In models such as Transformers, which operate on sequential data, positional embeddings play a crucial role in capturing the order and relative positions of elements in the sequence.
Instead of relying solely on fixed positional encodings (e.g., sine or cosine functions), learned positional embeddings introduce additional trainable parameters that can adaptively capture the sequential patterns present in the data. These embeddings are typically added to the input embeddings or intermediate representations of the model.

```
import tensorflow as tf

class PositionalEmbeddings(tf.keras.layers.Layer):
    """
    PositionalEmbeddings layer.

    This layer generates positional embeddings based on input IDs.
    It uses an Embedding layer to map position IDs to position embeddings.

    Args:
        config (object): Configuration object containing parameters.
    """

    def __init__(self, config, **kwargs):
        super(PositionalEmbeddings, self).__init__(**kwargs)
        self.supports_masking = True
        self.positional_embeddings = tf.keras.layers.Embedding(
            input_dim=config.max_position_embeddings, output_dim=config.hidden_size,
            mask_zero=True
        )

    def call(self, input_ids):
        """
        Generate positional embeddings.

        Args:
            input_ids (tf.Tensor): Input tensor containing token IDs.

        Returns:
            tf.Tensor: Positional embeddings tensor of shape (batch_size, seq_length, hidden_size).
        """
        seq_length = input_ids.shape[1]
        position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]
        position_embeddings = self.positional_embeddings(position_ids)
        return position_embeddings

    def get_config(self):
        """
        Get the layer configuration.

        Returns:
            dict: Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "positional_embeddings": self.positional_embeddings,
        })
        return config

```

By allowing the model to learn the positional representations, the learned positional embeddings enable the model to capture complex dependencies and patterns specific to the input sequence. The model can adapt its attention and computation based on the relative positions of the elements, which can be beneficial for tasks that require a strong understanding of the sequential nature of the data.

### Embedding layer

```

```

### Encoder
    The encoder is composed of multiple encoder layers that are stacked together. Each encoder layer takes a sequence of embeddings as input and processes them through two sublayers: a `multi-head self-attention` layer and a `fully connected feed-forward` layer. The output embeddings from each encoder layer maintain the same size as the input embeddings. The primary purpose of the encoder stack is to modify the input embeddings in order to create representations that capture contextual information within the sequence. For instance, if the words "keynote" or "phone" are in proximity to the word "apple," the encoder will adjust the embedding of "apple" to reflect more of a "company-like" context rather than a "fruit-like" one.

    Each of these sublayers also uses skip connections and layer normalization, which are standard tricks to train deep neural networks effectively. But to truly understand what makes a transformer work, we have to go deeper. Let’s start with the most important building block: the self-attention layer.

#### Self-Attention

    Self-attention, also known as intra-attention, is a mechanism in the Transformer architecture that allows an input sequence to attend to other positions within itself. It is a key component of both the encoder and decoder modules in Transformers. In self-attention, each position in the input sequence generates three vectors: Query (Q), Key (K), and Value (V). These vectors are linear projections of the input embeddings. The self-attention mechanism then computes a weighted sum of the values (V) based on the similarity between the query (Q) and key (K) vectors. The weights are determined by the dot product between the query and key vectors, followed by an application of the softmax function to obtain the attention distribution. This attention distribution represents the importance or relevance of each position to the current position.

The weighted sum of values, weighted by the attention distribution, is the output of the self-attention layer. This output captures the contextual representation of the input sequence by considering the relationships and dependencies between different positions. The self-attention mechanism allows each position to attend to all other positions, enabling the model to capture long-range dependencies and contextual information effectively.

One common implementation of self-attention, known as **scaled dot-product attention**, is widely used and described in the Vaswani et al. paper. This approach involves several steps to calculate the attention scores and update the token embeddings:

1. The token embeddings are projected into three vectors: query, key, and value.
2. Attention scores are computed by measuring the similarity between the query and key vectors using the dot product. This is efficiently achieved through matrix multiplication of the embeddings. Higher dot product values indicate stronger relationships between the query and key vectors, while low values indicate less similarity. The resulting attention scores form an n × n matrix, where n represents the number of input tokens.
3. To ensure stability during training, the attention scores are scaled by a factor to normalize their variance. Then, a softmax function is applied to normalize the column values, ensuring they sum up to 1. This produces the attention weights, which also form an n × n matrix.
4. The token embeddings are updated by multiplying them with their corresponding attention weights and summing the results. This process generates an updated representation for each embedding, taking into account the importance assigned to each token by the attention mechanism.

First of all let's prepare an input on which to test all the building blocks of the model:

```
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel
from transformers import TFAutoModel

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, from_pt=True)
model = TFAutoModel.from_pretrained(model_ckpt, from_pt=True)
token_emb_layer = tf.keras.layers.Embedding(model.config.vocab_size,model.config.hidden_size)

text = "time flies like an arrow"
inputs = tokenizer(text, add_special_tokens=False, return_tensors="tf")
inputs_embeds = token_emb_layer(inputs.input_ids)
tf.shape(inputs_embeds)
# <tf.Tensor: shape=(3,), dtype=int32, numpy=array([  1,   5, 768], dtype=int32)>
```

```
import tensorflow as tf

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Calculates scaled dot-product attention.

    Args:
        query: Query tensor (bs, len_q, dim).
        key: Key tensor (bs, len_k, dim).
        value: Value tensor (bs, len_v, dim).
        mask: Padding mask tensor (bs, len_) or None.

    Returns:
        Updated value embeddings after applying attention mechanism.
    """
    att_scores = tf.matmul(query, tf.transpose(key, perm=[0, 2, 1])) / tf.math.sqrt(tf.cast(tf.shape(query)[-1], tf.float32))

    if mask is not None:
        mask = tf.expand_dims(mask, axis=1)
        att_scores = tf.where(mask == 0, -1e9, att_scores)
        print(att_scores)

    att_weights = tf.nn.softmax(att_scores, axis=-1)
    n_value = tf.matmul(att_weights, value)

    return n_value
```

Testing:

```
scaled_dot_product_attention(inputs_embeds, inputs_embeds, inputs_embeds)

"""
<tf.Tensor: shape=(1, 5, 768), dtype=float32, numpy=
array([[[ 0.01556429,  0.00374883,  0.02096719, ..., -0.00231908,
         -0.00910157, -0.01922644],
        [ 0.0155272 ,  0.00371306,  0.0210163 , ..., -0.00231569,
         -0.00911147, -0.01903477],
        [ 0.0154297 ,  0.00385697,  0.02080077, ..., -0.00254198,
         -0.00929157, -0.0190092 ],
        [ 0.01515524,  0.00379813,  0.02099808, ..., -0.00259563,
         -0.00930912, -0.01920566],
        [ 0.01548781,  0.00350432,  0.02083553, ..., -0.00240597,
         -0.00937845, -0.01906679]]], dtype=float32)>
"""
```

#### Multi-headed attention

    In our simple example, we only utilized the embeddings in their original form to calculate attention scores and weights, but that’s far from the whole story. In practical applications, the self-attention layer employs three separate linear transformations on each embedding to generate query, key, and value vectors. These transformations project the embeddings and introduce their own unique learnable parameters. This enables the self-attention layer to concentrate on various semantic aspects of the sequence.

Furthermore, there is a clear advantage to incorporating multiple sets of linear projections, each representing an attention head. But why is it necessary to have more than one attention head? The reason is that when using just a single head, the softmax tends to focus primarily on one aspect of similarity.

By introducing multiple attention heads, the model gains the ability to simultaneously focus on multiple aspects. For instance, one head can attend to subject-verb interactions, while another head can identify nearby adjectives. This multi-head approach empowers the model to capture a broader range of semantic relationships within the sequence, enhancing its understanding and representation capabilities.

```

class AttentionHead(tf.keras.layers.Layer):
    """
    Attention head implementation.

    Args:
        head_dim: Dimensionality of the attention head.

    Attributes:
        head_dim: Dimensionality of the attention head.
        query_weights: Dense layer for query projection.
        key_weights: Dense layer for key projection.
        value_weights: Dense layer for value projection.
    """

    def __init__(self, head_dim, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True  # Enable masking support
        self.head_dim = head_dim
        self.query_weights = tf.keras.layers.Dense(head_dim)
        self.key_weights = tf.keras.layers.Dense(head_dim)
        self.value_weights = tf.keras.layers.Dense(head_dim)
        

    def call(self, hidden_state, mask = None):
        """
        Applies attention mechanism to the input hidden state.

        Args:
            hidden_state: Hidden state tensor (bs, len, dim).
            mask: Padding mask tensor (bs, len, len) or (bs, 1, len) or None.

        Returns:
            Updated hidden state after applying attention mechanism.
        """
        query = self.query_weights(hidden_state)
        key = self.key_weights(hidden_state)
        value = self.value_weights(hidden_state)

        attention_scores = scaled_dot_product_attention(query, key, value, mask = mask)
        return attention_scores


    def get_config(self):
        """
        Returns the configuration of the attention head layer.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "head_dim": self.head_dim,
            "query_weights": self.query_weights,
            "key_weights": self.key_weights,
            "value_weights": self.value_weights,
        })
        return config
```

Here we’ve initialized three independent linear layers that apply matrix multiplication to the embedding vectors to produce tensors of shape [batch_size, seq_len, head_dim], where head_dim is the number of dimensions we are projecting into. Although head_dim does not have to be smaller than the number of embedding dimensions of the tokens (embed_dim), in practice it is chosen to be a multiple of embed_dim so that the computation across each head is constant. For example, BERT has 12 attention heads, so the dimension of each head is 768/12 = 64. Now that we have a single attention head, we can concatenate the outputs of each one to implement the full multi-head attention layer:

```


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention layer implementation.

    Args:
        config: Configuration object containing hyperparameters.
    
    Attributes:
        supports_masking: Boolean indicating if the layer supports masking.
        hidden_size: Dimensionality of the hidden state.
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        attention_heads: List of AttentionHead layers.
        fc: Fully connected layer for final projection.

    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.attention_heads = [AttentionHead(self.head_dim) for _ in range(self.num_heads)]
        self.fc = tf.keras.layers.Dense(config.hidden_size)

    def call(self, hidden_state, mask=None):
        """
        Applies multi-head attention mechanism to the input hidden state.

        Args:
            hidden_state: Hidden state tensor (bs, len, dim).
            mask: Padding mask tensor (bs, len) or None.

        Returns:
            Updated hidden state after applying multi-head attention mechanism.
        """
        attention_outputs = [attention_head(hidden_state, mask=mask) for attention_head in self.attention_heads]
        hidden_state = tf.concat(attention_outputs, axis=-1)
        hidden_state = self.fc(hidden_state)
        return hidden_state

    def get_config(self):
        """
        Returns the configuration of the multi-head attention layer.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "attention_heads": self.attention_heads,
            "fc": self.fc,
        })
        return config
```

Notice that the concatenated output from the attention heads is also fed through a final linear layer to produce an output tensor of shape [batch_size, seq_len, hidden_dim] that is suitable for the feed-forward network downstream.

**Testing:**

```
multihead_attn = MultiHeadAttention(config)
attn_output = multihead_attn(inputs_embeds)
attn_output.shape
# TensorShape([1, 5, 768])
```

#### The Feed-Forward Layer and Normalization

    The feed-forward sublayer in both the encoder and decoder modules can be described as a simple two-layer fully connected neural network. However, its operation differs from a standard network in that it treats each embedding in the sequence independently rather than processing the entire sequence as a single vector. Because of this characteristic, it is often referred to as a **position-wise feed-forward layer**.

In the literature, a general guideline suggests setting the hidden size of the first layer to be four times the size of the embeddings. Additionally, a GELU activation function is commonly used in this layer. It is believed that this particular sublayer contributes significantly to the model's capacity and memorization abilities. Consequently, when scaling up the models, this layer is often a focal point for adjustment and expansion.

```
class FeedForward(tf.keras.layers.Layer):
    """
    Feed-forward layer implementation.

    Args:
        config: Configuration object containing hyperparameters.

    Attributes:
        supports_masking: Boolean indicating if the layer supports masking.
        fc1: First dense layer.
        fc2: Second dense layer.
        dropout: Dropout layer.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.fc1 = tf.keras.layers.Dense(config.intermediate_fc_size, activation=tf.keras.activations.gelu)
        self.fc2 = tf.keras.layers.Dense(config.hidden_size)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_state, training=False):
        """
        Applies feed-forward transformation to the input hidden state.

        Args:
            hidden_state: Hidden state tensor (batch_size, sequence_length, hidden_size).
            training: Boolean indicating whether the model is in training mode or inference mode.

        Returns:
            Updated hidden state after applying feed-forward transformation.
        """
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.dropout(hidden_state, training=training)
        hidden_state = self.fc2(hidden_state)
        return hidden_state

    def get_config(self):
        """
        Returns the configuration of the feed-forward layer.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "fc1": self.fc1,
            "fc2": self.fc2,
            "dropout": self.dropout,
        })
        return config
```
It is important to note that when using a feed-forward layer like `dense`, it is typically applied to a tensor with a shape of `(batch_size, input_dim)`. In this case, the layer operates independently on each element of the batch dimension. This applies to all dimensions except for the last one. Therefore, when we pass a tensor with a shape of `(batch_size, seq_len, hidden_dim)`, the feed-forward layer is applied to each token embedding of the batch and sequence separately, which aligns perfectly with our desired behavior.

**Adding Layer Normalization:**

    When it comes to placing layer normalization in the encoder or decoder layers of a transformer, there are two main choices that have been widely adopted in the literature. The first choice is to apply layer normalization before each sub-layer, which includes the self-attention and feed-forward sub-layers. This means that the input to each sub-layer is normalized independently , it's called **Pre layer normalization**. The second choice is to apply layer normalization after each sub-layer, which means that the normalization is applied to the output of each sub-layer, it's called **Post layer normalization**. Both approaches have their own advantages and have been shown to be effective in different transformer architectures. The choice of placement often depends on the specific task and architecture being used.

#### Encoder layer

Now that we've built all the main parts of the encoder layer, we'll put them together to build it:
```
class Encoder(tf.keras.layers.Layer):
    """
    Encoder layer of the Transformer model.

    Args:
        config: Configuration object containing hyperparameters.

    Attributes:
        supports_masking: Boolean indicating if the layer supports masking.
        multihead_attention: MultiHeadAttention layer for attention mechanism.
        norm1: LayerNormalization layer for the first normalization step.
        norm2: LayerNormalization layer for the second normalization step.
        feed_forward: FeedForward layer for the feed-forward network.
        dropout: Dropout layer for regularization.

    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True 
        self.multihead_attention = MultiHeadAttention(config)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.feed_forward = FeedForward(config)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        

    def call(self, hidden_state, mask=None, training=False):
        """
        Applies the encoder layer to the input hidden state.

        Args:
            hidden_state: Hidden state tensor (bs, len, dim).
            mask: Padding mask tensor (bs, len) or None.
            training: Boolean indicating whether the model is in training mode or inference mode.

        Returns:
            Updated hidden state after applying the encoder layer.

        """
        attention_output = self.multihead_attention(hidden_state)  # Apply multi-head attention
        hidden_state = self.norm1(attention_output + hidden_state)  # Add skip connection and normalize
        feed_forward_output = self.feed_forward(hidden_state)  # Apply feed-forward layer
        hidden_state = self.norm2(feed_forward_output + hidden_state)  # Add skip connection and normalize
        hidden_state = self.dropout(hidden_state, training=training)  # Apply dropout
        return hidden_state

    def get_config(self):
        """
        Returns the configuration of the encoder layer.

        Returns:
            Configuration dictionary.

        """
        config = super().get_config()
        config.update({
            "multihead_attention": self.multihead_attention,
            "norm1": self.norm1,
            "norm2": self.norm2,
            "feed_forward": self.feed_forward,
            "dropout": self.dropout,
        })
        return config
```

We’ve now implemented our very first transformer encoder layer from scratch!




