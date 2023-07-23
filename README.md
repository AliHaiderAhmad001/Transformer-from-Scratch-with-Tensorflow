# Neural-Machine-Translator with Transformers

The task of translating text from one language into another is a common task. In this repo I'm going to construct and demonstrate a naive model for an English to French translation task using TensorFlow, which is inspired by the model presented in the "Attention Is All You Need" paper. With this repo, I aim to build a reference that may be useful to newcomers or someone who wants to understand or build Transformers from scratch.

تعتبر مهمة ترجمة النص من لغة إلى لغة أخرى من المهام الشائعة. سأقوم في هذا الريبو ببناء وشرح نموذج بسيط لمهمة الترجمة من اللغة الإنجليزية إلى اللغة الفرنسية باستخدام تنسرفلو، وهو مستلهم من النموذج الذي تم تقديمه في ورقة "الانتباه هو كل ماتحتاجه". أهدف من خلال هذا المشروع إلى بناء مرجع قد يفيد الوافدين الجدد أو أحدًا ما يريد أن يفهم أو يبني المحولات من الصفر.
<br>
<br>
<br>

## Transformer architecture

The Transformer architecture is a popular model architecture used for various natural language processing tasks, including machine translation, text generation, and language understanding. It was introduced by Vaswani et al. in the paper "Attention Is All You Need". The Transformer architecture consists of two main components: the encoder and the decoder. Both the encoder and decoder are composed of multiple layers of self-attention and feed-forward neural networks. The key idea behind the Transformer is the use of self-attention mechanisms, which allow the model to focus on different parts of the input sequence when generating the output. In the following figure a high-level overview of the Transformer architecture.

المحولات هي بنية نموذجية شائعة تستخدم في العديد من مهام معالجة اللغة الطبيعية، بما في ذلك الترجمة الآلية وتوليد النص وفهم اللغة. تم تقديمه بواسطة Vaswani et al. في مقالة "الاهتمام هو كل ما تحتاجه". تتكون بنية المحول من مكونين رئيسيين: المشفر ووحدة فك التشفير. يتكون كل من المشفر ومفكك التشفير من طبقات متعددة من الاهتمام الذاتي والشبكات العصبية ذات التغذية الأمامية. الفكرة الرئيسية وراء المحول هي استخدام آليات الانتباه الذاتي، والتي تسمح للنموذج بالتركيز على أجزاء مختلفة من سلسلة الدخل عند إنشاء سلسلة الخرج. في الشكل التالي نظرة عامة عالية المستوى على بنية المحولات.
<br>
<br>
![Transformer architecture](imgs/transformer_arch.png "Transformer architecture")
<br>
<br>
* **Encoder.** The input sequence is first passed through an embedding layer, which maps each token to a continuous vector representation -followed by positional encoding to provide information about the position of tokens in the sequence. The embedded input is then processed by a stack of identical encoder layers. Each encoder layers consists of a multi-head self-attention mechanism, followed by a feed-forward neural network. The self-attention mechanism allows the *encoder* to attend to different positions in the input sequence and capture dependencies between tokens. The output of the encoder is a set of encoded representations for each token in the input sequence.

تُمرّر سلسلة الدخل إلى طبقة تضمين تقوم بربط كل وحدة نصية (كلمات في حالتنا) بشعاع رقمي يُمثّلها مكون من قيم مستمرة (أعداد حقيقية). يتبع ذلك ترميّز موضعي يهدف إلى تقديم معلومات حول مواقع الوحدات النصية ضمن سلسلة الدخل. تجري معالجة هذه التضمينات بعد ذلك من خلال طبقات المشفّر المتماثلة. كل طبقة تشفير تتألف من طبقة انتباه ذاتي متعدد الرؤوس متبوعة بطبقة تغذية أمامية. تسمح آلية الانتباه الذاتي للمشفّر بمشاهدة المواقع المختلفة لسلسلة الدخل، والتقاط التبعيات بين الوحدات النصية. أي بمعنى آحر، تسمح له بمعرفة مدى ارتباط كل كلمة مع الكلمات الأخرى. خرج المشفّر هو تمثيّل جديد لكل وحدة نصيّة من سلسلة الدخل.

* **Decoder.** The decoder takes the encoded representations from the encoder as input, along with the target sentence. Similar to the encoder, the input to the decoder is first passed through an embedding layer, followed by positional encoding. The decoder also consists of a stack of identical decoder layers. Each decoder layer has two self-attention mechanisms: a masked multi-head self-attention mechanism to prevent the decoder from looking ahead, and a cross-attention mechanism that allows the decoder to attend to the encoder's output. The decoder generates the output sequence.

تأخذ وحدة فك التشفير التمثيلات من المشفّر كدخل، جنبًا إلى جنب مع الجملة الهدف. كما في وحدة المُشفّر؛ تُمرّر سلسلة الدخل إلى طبقة تضمين تقوم بربط كل وحدة نصية (كلمات في حالتنا) بشعاع رقمي يُمثّلها مكون من قيم مستمرة (أعداد حقيقية). يتبع ذلك ترميّز موضعي. تحتوي كل طبقة من طبقات وحدة فك التشفير على آليتين للانتباه الذاتي: آلية الانتباه الذاتي متعدد الرؤوس المُقنّعة لمنع وحدة فك التشفير من النظر إلى الأمام، وآلية الانتباه المتقاطع الذي يسمح لوحدة فك التشفير بالوصول إلى خرج المشفّر. تقوم وحدة فك التشفير بتوليد سلسلة الخرج (الجملة بعد الترجمة).

* **Positional Encoding.** Positional encoding is used to inject information about the position of tokens into the model. The positional encoding is added to the embedded input and provides the model with the positional information necessary for capturing sequential relationships.

يُستخدم الترميز الموضعي لإضافة معلومات المواقع للوحدات النصية إلى النموذج (بدونها لايمكن للنموذح فهم ترتيب الكلمات في الجملة).

* **Final Linear Layer.** The output of the decoder is passed through a linear layer followed by a Softmax activation function to produce the probability distribution over the target vocabulary.

يتم تمرير خرج وحدة فك التشفير إلى طبقة خطية متبوعة بدالة تنشيط سوفتماكس لإنتاج التوزيع الاحتمالي على المفردات المستهدفة.
<br>
<br>

The following figure shows encoder-decoder architecture of the transformer, with the encoder shown in the upper half of the figure and the decoder in the lower half. It is worth saying that this figure represents the the "inference phase". Here it should be noted that the decoder’s mechanism in the inference phase is slightly different in the training phase, and we will understand that later.

يوضّح الشكل التالي بنية المُشفّر-فك التشفير في المحوّل، حيث يبيّن النصف العلوي من الشكل بنية المُشفّر، ووحدة فك التشفير في النصف السفلي. تجدر الإشارة إلا أن هذا الشكل يمثل "مرحلة الاستدلال"، حيث أن آلية فك التشفير في مرحلة الاستدلال تختلف قليلاً عن مرحلة التدريب، وسنفهم ذلك لاحقًا عند مناقشة مرحلة التدريب والاستدلال.
<br>
<br>
![Encoder-decoder examble](imgs/Transformer.png "Encoder-decoder examble")
<br>
<br>

We’ll look at each of this components in details later, but we can already see a few things in Figure above that characterize the Transformer architecture. The input text is tokenized and converted to token embeddings. Since the attention mechanism is not aware of the relative positions of the tokens, we need a way to inject some information about token positions into the input to model the sequential nature of text. The token embeddings are thus combined with positional embeddings that contain positional information for each token. The encoder is composed of a stack of encoder layers or "blocks", which is analogous to stacking convolutional layers in computer vision. The same is true of the decoder, which has its own stack of decoder layers. The encoder's output is fed to each decoder layer, and the decoder then generates a prediction for the most probable next token in the sequence. The output of this step is then fed back into the decoder to generate the next token, and so on until a special end-of-sequence *EOS* token is reached. In the example from Figure above, imagine the decoder has already predicted "Die" and *Zeit*. Now it gets these two as an input as well as all the encoder's outputs to predict the next token, *fliegt*. In the next step the decoder gets *fliegt* as an additional input. We repeat the process until the decoder predicts the *EOS* token or we reached a maximum length. Here are the steps to build an English to French translation model using the Transformers architecture:


نلقي نظرة على كل مكون بالتفصيل قريبًا، ولكن يمكننا رؤية بعض الأشياء في الشكل أعلاه والتي تُميّز بنية المحولات. يتم تقطيع النص المُدخل إلى وحدات نصية (يمكن اعتبارها كلمات) ثم يتم ربط كل وحدة نصية بشعاع تضمين يمثّلها. بما أن آلية الانتباه لاتستطيع التعرّف على مواقع الوحدات النصية، فإننا بحاجة إلى طريقةٍ ماتمكننا من إضافة بعض المعلومات التي تعبر عن مواقع هذه الوحدات إلى تمثيلات الوحدات النصية السابقة. يتم ذلك من خلال طبقة الترميز الموضعي، حيث يتم ربط كل موقع ضمن التسلسل بشعاع يُضاف إلى تمثيل الوحدة النصية الموافقة لذلك الموقع. يتألف المُشفّر من عدة طبقات متماثلة من حيث البنية ومكدسة فوق بعضها بطريقة مشابهة لتكديس طبقات *CNN*، كل منها تدعى "طبقة تشفير" أو "كِتل". يتم تمرير تمثيلات الوحدات النصية إلى هذه الكتل، ويكون خرجها تمثيل جديد أكثر قوة للوحدات النصية. يتم تمرير خرج وحدة المُشفّر (تمثيلات الوحدات النصية لجملة الدخل) إلى كل طبقة فك تشفير في وحدة فك التشفير (طبقة فك التشفير تتألف من عدة طبقات فك تشفير، أي بشكل مشابه لوحدة التشفير). تقوم وحدة فك التشفير بتوليد توقع يُمثّل الوحدة النصية التالية في جملة الهدف -الأكثر رجوحًا. تستمر عملية التوليد هذه وصولًا إلى رمز نهاية السلسلة *EOS*. في المثال الموضّح في الشكل أعلاه، تخيل أن وحدة فك التشفير توقعت كلمة *Die* وكلمة *Zeit*. الآن سيتم أخذ هذه الكلمات كدخل إلى وحدة فك التشفير جنبًا إلى جنب مع خرج المُشفّر لتوقع الوحدة النصية التالية والتي هي *fliegt*. في الخطوة التالية سيتم استخدام الكلمات الجديدة جنبًا إلى جنبًا مع الكلمات السابقة وخرج المشفر لتوليد الكلمة التالية. يتم تكؤار هذه العملية حتى يتم توقع رمز نهاية الجملة EOS. فيما يلي خطوات بناء نموذج ترجمة من الإنجليزية إلى الفرنسية باستخدام بنية المحولات.


## Data preparation

We'll be working with an [English-to-French translation dataset](https://ankiweb.net/shared/decks/french) (مجموعة البيانات التي سنعمل معها):

```
import pathlib
import tensorflow as tf
text_file = tf.keras.utils.get_file(
    fname="fra-eng.zip",
    origin="http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip",
    extract=True,
)
text_file = pathlib.Path(text_file).parent/"fra.txt"
print(text_file) # /root/.keras/datasets/fra.txt
```

The dataset we're working on consists of 167,130 lines. Each line consists of the original sequence (the sentence in English) and the target sequence (in French). Now, normalize dataset (French and English sentence) then prepend the token `[start]` and append the token `[end]` to the French sentence:

تتكون مجموعة البيانات التي نعمل عليها من 167130 سطرًا. يتكون كل سطر من التسلسل الأصلي (الجملة باللغة الإنجليزية) والتسلسل المستهدف (بالفرنسية). نُجري الآن تقييّسًا لمجموعة البيانات (الجملة الفرنسية والإنجليزية) ثم نُرفق الوحدة النصية المُخصّصة `[start]` في البداية والوحدة النصية المُخصّصة `[end]` في النهاية للجمل الفرنسية فقط:

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
    # converts characters to their standardized forms (e.g., converting full-width characters to half-width)
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

# let's look at the data
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
    
Now, we need to write a function that associates each token with a unique integer number representing it to get what is called a *Tokens_IDs*. Fortunately, there is a layer in TensorFlow called [`TextVectorization`](https://keras.io/api/layers/preprocessing_layers/core_preprocessing_layers/text_vectorization/) that makes life easier for us. We'll use two instances of the TextVectorization to vectorize the text data (one for English and one for Spanish). First of all, let's split the sentence pairs into a training set, a validation set, and a test set:

 نحتاج الآن إلى كتابة دالة تربط كل وحدة نصية برقم صحيح فريد يمثلها للحصول على ما يسمى *Tokens_IDs*. لحسن الحظ هناك طبقة في TensorFlow تسمى [`TextVectorization`] (https://keras.io/api/layers/preprocessing_layers/core_preprocessing_layers/text_vectorization/) تجعل الأمور أسهل بالنسبة لنا. سنستخدم كائنين من `TextVectorization` لتوجيه (التحويل لمصفوفات) البيانات النصية (أحدهما للغة الإنجليزية والآخر للغة الإسبانية). بدايةً نقسم أزواج الجمل إلى مجموعة تدريب ومجموعة مراقبة ومجموعة اختبار:

```
random.shuffle(text_pairs)
max_len = max([max([len(x[0].split()), len(x[1].split())]) for x in text_pairs])
num_val_samples = int(0.15 * len(text_pairs)) # 15% 
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

print(f"{max_len} maximum length")
print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")
```

```
Output:
60 maximum length
167130 total pairs
116992 training pairs
25069 validation pairs
25069 test pairs
```
    
Now we'll do Vectorization (نطبق عملية التوجيه):

```
from tensorflow.keras.layers import TextVectorization

vocab_size_en = 14969
vocab_size_fr = 29219
seq_length = 256

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

Now we can save what we do for the subsequent steps (Optional):

يمكنك حفظ العمل الذي قمت به حتى الآن كما يلي (اختياري):

```
with open("vectorize.pickle", "wb") as fp:
    data = {
        "train": train_pairs,
        "val":   val_pairs,
        "test":  test_pairs,
        "engvec_config":  eng_vectorizer.get_config(),
        "engvec_weights": eng_vectorizer.get_weights(),
        "fravec_config":  fra_vectorizer.get_config(),
        "fravec_weights": fra_vectorizer.get_weights(),
    }
    pickle.dump(data, fp)
```

We can open the files again using (فتح الملفات مرة أخرى):
```
with open("vectorize.pickle", "rb") as fp:
    data = pickle.load(fp)

train_pairs = data["train"]
val_pairs = data["val"]
test_pairs = data["test"]
eng_vectorizer = TextVectorization.from_config(data["engvec_config"])
eng_vectorizer.set_weights(data["engvec_weights"])
fra_vectorizer = TextVectorization.from_config(data["fravec_config"])
fra_vectorizer.set_weights(data["fravec_weights"])
```

Now we have to define how we will pass the data to the model. There are several ways to do this:
* Present the data as a NumPy array or a tensor (Faster, but need to load all data into memory).
* Create a Python generator function and let the loop read data from it (Fetched from the hard disk when needed, rather than being loaded all into memory).
* Use the `tf.data` dataset(Our choice). The general benefits of using the `tf.data` dataset are flexibility in handling the data and making feeding the model data more efficient and fast. But what is `tf.data` (In brief)? `tf.data` is a module in TensorFlow that provides tools for building efficient and scalable input pipelines for machine learning models. It is designed to handle large datasets, facilitate data preprocessing, and enable high-performance data ingestion for training and evaluation. Using tf.data, you can build efficient and scalable input pipelines for training deep learning models. Here are some important functions:
    * `shuffle(n)`: Randomly fills a buffer of data with `n` data points and randomly shuffles the data in the buffer.
    * `batch(n)`: Generate batches of the dataset, each of size n.
    * `prefetch(n)`: to keep n batches/elements in memory ready for the training loop to consume.
    * `cache(): Efficiently caches the dataset for faster subsequent reads.
    * `map(func)`: Applying a transform (function) on data batches.
    * [You can read more here](https://pyimagesearch.com/2021/06/14/a-gentle-introduction-to-tf-data-with-tensorflow/).
    * [ِAnd here](https://stackoverflow.com/questions/76414594/shuffle-the-batches-in-tensorflow-dataset/76443517#76443517).


الآن علينا تحديد كيف سنقوم بتمرير البيانات إلى النموذج. هناك عدة طرق للقيام بذلك:
* تقديم البيانات كمصفوفة NumPy أو Tensor (أسرع ولكن يجب تحميل جميع البيانات في الذاكرة -عبء-).
* إنشاء دالة توليّد والسماح لحلقة بقراءة البيانات منها (يمكن استرجاعها من القرص الثابت عند الحاجة، بدلاً من تحميلها جميعًا في الذاكرة).
* استخدام `tf.data` (اختيارنا). تتمثل المزايا العامة لاستخدام `tf.data` في المرونة في التعامل مع البيانات وجعل إمداد البيانات إلى النموذج أكثر كفاءة وسرعة. ولكن ما هي `tf.data` (باختصار)؟ `tf.data` هو وحدة برمجية في تنسرفلو توفر أدواتًا لبناء أنابيب إدخال فعاّلة وقابلة للتوسع لنماذج التعلم الآلي. تم تصميمها للتعامل مع مجموعات البيانات الكبيرة، وتيسير معالجة البيانات، وتمكين استيعاب البيانات مع أداء عالي في التدريب والتقييم (يشير استيعاب البيانات إلى عملية التحميل والمعالجة المسبقة وتغذية البيانات في النموذج). يمكنك باستخدام `tf.data` بناء أنابيب إدخال فعّالة وقابلة للتوسع لتدريب نماذج التعلم العميق. فيما يلي بعض الدوال الهامة:
   * الدالة `shuffle(n)`: يقوم عشوائيًا بملء مخزن مؤقت بـ `n` عينة بيانات ويعمل على خلط البيانات في المخزن.
   * الدالة `batch(n)`: يولد دُفعات من مجموعة البيانات، كل منها بحجم n.
   * الدالة `prefetch(n)`: للاحتفاظ ب n دُفعات/عناصر جاهزة في الذاكرة تمهيدًا لاستهلاكها في التكرار التالي لعملية التدريب.
   * الدالة `cache`: يقوم بتخبئة البيانات بكفاءة لتسريع القراءة في المرات التالية (اعتقد أنها تحاول عدم تكرار نفس عمليات المعالجة في كل مرة، أي تجعلها تُطبّق مرة واحدة).
   * الدالة `map(func)`: تطبيق تحويل (دالة) على البيانات.
  * [اقرأ أكثر](https://pyimagesearch.com/2021/06/14/a-gentle-introduction-to-tf-data-with-tensorflow/).
  * [ِوهنا أيضًا](https://stackoverflow.com/questions/76414594/shuffle-the-batches-in-tensorflow-dataset/76443517#76443517).
```
from tensorflow.data import AUTOTUNE

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


def make_dataset(pairs, batch_size=64):
    """
    Creates a dataset from pairs of English and French texts.

    Args:
        pairs: List of pairs containing English and French texts.
        batch_size: Batch size for the dataset.

    Returns:
        Formatted and preprocessed dataset.
    """
    eng_texts, fra_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    fra_texts = list(fra_texts)
    
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, fra_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(2048).prefetch(tf.data.experimental.AUTOTUNE).cache()

    return dataset

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

# let's take a look:
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


## Transformer Building Blocks
We will now build each component of the Transformer independently.
<br>
<br>
### Positional information

Positional encoding is a technique used in transformers to incorporate the positional information of words or tokens into the input embeddings. Since transformers don't have any inherent notion of word order, positional encoding helps the model understand the sequential order of the input sequence. In transformers, positional encoding is typically added to the input embeddings before feeding them into the encoder or decoder layers. The positional encoding vector is added element-wise to the token embeddings, providing each token with a unique position-dependent representation. There are three common types of positional encodings used in transformers:

* **Sinusoidal positional encoding.** Is based on sine and cosine functions with different frequencies for each dimension of the embedding.
* **Learned Positional Embeddings.** Instead of using fixed sinusoidal functions, learned positional embeddings are trainable parameters that are optimized during the model training. These embeddings can capture position-specific patterns and dependencies in the input sequence.
* **Relative Positional Encodings.** In addition to absolute positional information, relative positional encodings take into account the relative distances or relationships between tokens. They can be used to model the dependencies between tokens at different positions in a more explicit manner.

The choice of positional encoding depends on the specific task, dataset, and model architecture. Sinusoidal positional encoding is widely used and has shown good performance in various transformer-based models. However, experimenting with different types of positional encodings can be beneficial to improve the model's ability to capture positional information effectively.

In this section we will investigate the two approaches: **Learned Positional Embeddings** and ٍ **Sinusoidal positional encoding**. However, it may not be necessary to use complex positional encoding methods. The standard sinusoidal positional encoding used in the original Transformer model can still work well.


التشفير الموضعي هو تقنية تُستخدم في المُحوّلات لدمج المعلومات الموضعية للكلمات أو الرموز في تضمينات المدخلات. نظرًا لعدم وجود أي مفهوم ضمني لترتيب الكلمات في المُحوّلات، يُساعد التشفير الموضعي النموذج على فهم التسلسل الزمني لسلسلة الوحدات النصية المُدخلة. عادةً ما يتم إضافة التشفير الموضعي إلى التضمينات قبل تغذيتها إلى طبقات المُشفِّر أو المُفَكِّك. يتم إضافة متجه التشفير الموضعي بالجمع عنصرًا بعنصر إلى التضمينات، مما يمنح كل وحدة نصية تمثيلًا فريدًا للموضع.

تُستخدم ثلاثة أنواع شائعة من التشفير الموضعي في المُحوّلات:

* **التشفير الموضعي الجيبي.** يعتمد على دوال الجيب وجيب التمام بترددات مختلفة لكل بُعد من أبعاد التضمين.

* **تضمينات الموضع المُتعلّمة:** بدلاً من استخدام وظائف ساينوسية ثابتة، تكون تضمينات الموضع المتعلمة معاملات قابلة للتدريب ويتم تحسينها أثناء تدريب النموذج. يمكن لهذه التضمينات التقاط أنماط وتبعيات تعتمد على الموضع في المتوالية المدخلة.

* **تشفير الموضع النسبي:** بالإضافة إلى المعلومات الموضعية المطلقة، يأخذ تشفير الموضع النسبي في اعتباره المسافات النسبية أو العلاقات بين الوحدات النصية. يمكن استخدامها لنمذجة التبعيات بين الوحدات النصية في مواضع مختلفة بطريقة صريحة.

تعتمد اختيار الترميز الموضعي على المهمة المحددة وقاعدة البيانات وهندسة النموذج المستخدم. يُستخدم ترميز الموضع الساينوسي على نطاق واسع وأظهر أداءً جيدًا في مختلف النماذج المعتمدة على النماذج المحوَّلة. ومع ذلك، يمكن أن تكون التجارب على أنواع مختلفة من الترميز الموضعي مفيدة لتحسين قدرة النموذج على التقاط المعلومات الموضعية بفعالية.

نتحقق من نهجين: **تضمينات الموضع المتعلمة** و**التشفير الموضعي الجيبي**. غالبًا ما لا يكون من الضروري استخدام طرق تشفير موضعية معقدة، فالتشفير الموضعي الجيبي القياسي المستخدم في النموذج الأصلي يعمل بشكل جيد.


#### Sinusoidal positional encoding

The most commonly used method for positional encoding in transformers is the sinusoidal positional encoding, as introduced in the "Attention Is All You Need" paper by Vaswani et al. The sinusoidal positional encoding is based on the idea that different positions can be represented by a combination of sine and cosine functions with different frequencies. The formula for the sinusoidal positional encoding is as follows:

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where `PE(pos, 2i)` represents the `i-th` dimension of the positional encoding for the token at position `pos`, and `d_model` is the dimensionality of the model.

الطريقة الأكثر استخدامًا لتشفير الموضع في المحولات هي التشفير الموضعي الساينوسي (الجيبي)، كما تم تقديمه في ورقة "الانتباه هو كل ما تحتاجه" بواسطة فاسواني وآخرون. يعتمد التشفير الموضعي الساينوسي على فكرة أنه يمكن تمثيل المواضع المختلفة باستخدام مجموعة من الدوال الجيبية والتكسيرية ذات الترددات المختلفة. تتمثل الصيغة الرسمية للتشفير الموضعي الساينوسي كما في المعادلة أعلاه، حيث يُمثل `PE(pos, 2i)` البُعد `i` للتشفير الموضعي للرمز في الموضع `pos`، و `d_model` هي بُعد النموذج.

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

Testing:

```
# Define the configuration
class Config:
    def __init__(self):
        self.sequence_length = 4
        self.hidden_size = 4
        self.frequency_factor = 10000
        
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
print(input_ids)
print(output_embeddings)
```

```
Outputs:
tf.Tensor([[2 0 0 0]], shape=(1, 4), dtype=int32)
tf.Tensor(
[[ 0.          0.          1.          1.        ]
 [ 0.84147096 -0.50636566  0.5403023   0.8623189 ]
 [ 0.9092974  -0.87329733 -0.4161468   0.48718765]
 [ 0.14112    -0.99975586 -0.9899925  -0.02209662]], shape=(4, 4), dtype=float32)
```

By using sinusoidal positional encoding, the model can differentiate between tokens based on their positions in the input sequence. This allows the transformer to capture sequential information and attend to different parts of the sequence appropriately. It's important to note that positional encoding is added as a fixed representation and is not learned during the training process. The model learns to incorporate the positional information through the attention mechanism and the subsequent layers of the transformer.

باستخدام التشفير الموضعي الجيبي، يمكن للنموذج التفريق بين الكلمات بناءً على مواقعها في تسلسل الإدخال. هذا يسمح للمحول بالتقاط المعلومات المتسلسلة والاهتمام بأجزاء مختلفة من التسلسل. من المهم ملاحظة أن التشفير الموضعي يُضاف كتمثيل ثابت ولا يتم تعلّمه أثناء عملية التدريب. يتعلم النموذج دمج المعلومات الموضعية من خلال آلية الانتباه والطبقات اللاحقة للمحول.

#### Learned Positional Embeddings

Learned positional embeddings refer to the practice of using trainable parameters to represent positional information in a sequence. In models such as Transformers, which operate on sequential data, positional embeddings play a crucial role in capturing the order and relative positions of elements in the sequence (but they do not explicitly capture the relative positions).
Instead of relying solely on fixed positional encodings (e.g., sine or cosine functions), learned positional embeddings introduce additional trainable parameters that can adaptively capture the sequential patterns present in the data. These embeddings are typically added to the input embeddings.

تشير التضمينات الموضعية المُتعلّمة إلى استخدام المعلمات القابلة للتدريب لتمثيل المعلومات الموضعية في تسلسل. في نماذج مثل المحوّلات، والتي تعمل على بيانات متسلسلة، تلعب التضمينات الموضعية دورًا مهمًا في التقاط الترتيب والمواضع النسبية للعناصر في التسلسل (لكنها لا تلتقط بوضوح المواضع النسبية).
بدلاً من الاعتماد فقط على التشفيرات الموضعية الثابتة (دوال الجيب أو جيب التمام)، تُقدّم التضمينات الموضعية المُتعلّمة معلمات قابلة للتدريب يمكنها التقاط الأنماط المتسلسلة الموجودة في البيانات بشكل تكيفي. تضاف هذه التضمينات عادةً إلى التضمينات المدخلة كما في حالة دوال الجيب.

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
        self.positional_embeddings = tf.keras.layers.Embedding(
            input_dim=config.max_position_embeddings, output_dim=config.hidden_size
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

Testing:

```
# Define the configuration
class Config:
    def __init__(self):
        self.sequence_length = 4
        self.hidden_size = 4
        self.frequency_factor = 10000
        self.max_position_embeddings = 4
        self.mask_zero = True
        

config = Config()

# Create an instance of the SinusoidalPositionalEncoding layer
positional_encoding_layer = PositionalEmbeddings(config)

# Create a sample input tensor with token IDs
batch_size = 1
seq_length = 4
input_ids = tf.random.uniform((batch_size, seq_length), maxval=config.sequence_length, dtype=tf.int32)

# Apply positional encodings
output_embeddings = positional_encoding_layer(input_ids)

# Print the output positional embeddings
print("Outputs:")
print(output_embeddings)
```
```
Outputs:
tf.Tensor(
[[[-0.02825731 -0.00217507  0.01578121 -0.01750519]
  [ 0.00112041 -0.03614271  0.03306187 -0.02413228]
  [ 0.00990455 -0.00736488  0.03470118 -0.02544773]
  [ 0.02571186 -0.02450178 -0.02327818  0.04356712]]], shape=(1, 4, 4), dtype=float32)
```

By allowing the model to learn the positional representations, the learned positional embeddings enable the model to capture complex dependencies and patterns specific to the input sequence. The model can adapt its attention and computation based on the relative positions of the elements, which can be beneficial for tasks that require a strong understanding of the sequential nature of the data.

 يُمكن من خلال السماح للنموذج بتعلم التمثيلات الموضعية، التقاط التبعيات والأنماط المعقدة الخاصة بتسلسل الإدخال. يمكن للنموذج تكييف انتباهه وحسابه بناءً على المواضع النسبية للعناصر، والتي يمكن أن تكون مفيدة للمهام التي تتطلب فهمًا قويًا للطبيعة المتسلسلة للبيانات.
 
<br>
<br>

### Embedding layer

Now we are going to build the Embeddings layer. This layer will take `inputs_ids` and associate them with primitive representations and add positional information to them.

الآن سنقوم ببناء طبقة التضمينات. ستأخذ هذه الطبقة *المُدخلات* وتربطها بالتمثيلات البدائية وتضيف إليها المعلومات الموضعية.

```
import tensorflow as tf

class Embeddings(tf.keras.layers.Layer):
    """
    Embeddings layer.

    This layer combines token embeddings with positional embeddings to create the final embeddings.

    Args:
        config (object): Configuration object containing parameters.
        vocab_size: Vocabulary size.

    Attributes:
        token_embeddings (tf.keras.layers.Embedding): Token embedding layer.
        PositionalInfo (tf.keras.layers.Layer): Positional information layer.
        dropout (tf.keras.layers.Dropout): Dropout layer for regularization.
        norm (tf.keras.layers.LayerNormalization): Layer normalization for normalization.
    """

    def __init__(self, config, vocab_size,  **kwargs):
        super(Embeddings, self).__init__(**kwargs)
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim= vocab_size, output_dim=config.hidden_size
        )
        if config.positional_information_type == 'embs':
            self.PositionalInfo = PositionalEmbeddings(config)
        elif config.positional_information_type == 'sinu':
            self.PositionalInfo = SinusoidalPositionalEncoding(config)

        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, input_ids, training=False):
        """
        Generate embeddings for input IDs.

        Args:
            input_ids (tf.Tensor): Input tensor containing token IDs.
            training (bool, optional): Whether the model is in training mode. Defaults to False.

        Returns:
            tf.Tensor: Embeddings tensor of shape (batch_size, seq_length, hidden_size).
        """
        positional_info = self.PositionalInfo(input_ids)
        x = self.token_embeddings(input_ids)
        x += positional_info
        x = self.norm(x)
        x = self.dropout(x, training=training)
        return x

    def compute_mask(self, inputs, mask=None):
        """
        Computes the mask for the inputs.

        Args:
            inputs (tf.Tensor): Input tensor.
            mask (tf.Tensor, optional): Mask tensor. Defaults to None.

        Returns:
            tf.Tensor: Computed mask tensor.
        """
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        """
        Get the layer configuration.

        Returns:
            dict: Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "token_embeddings": self.token_embeddings,
            "PositionalInfo": self.PositionalInfo,
            "dropout": self.dropout,
            "norm": self.norm,
        })
        return config
```

Testing:

```
# Define the configuration
class Config:
    def __init__(self):
        self.sequence_length = 4
        self.hidden_size = 4
        self.frequency_factor = 10000
        self.max_position_embeddings = 4
        self.positional_information_type = 'embs'
        self.source_vocab_size = 10
        self.target_vocab_size = 10
        self.hidden_dropout_prob = 0.1


config = Config()

# Create an instance of the SinusoidalPositionalEncoding layer
Embeddings_layer = Embeddings(configو vocab_size = 10)

# Create a sample input tensor with token IDs
batch_size = 1
seq_length = 4
input_ids = tf.random.uniform((batch_size, seq_length), maxval=config.sequence_length, dtype=tf.int32)

# Apply positional encodings
output_embeddings = Embeddings_layer(input_ids)

# Print the output positional embeddings
print("Outputs:")
print(output_embeddings)
"""
Outputs:
tf.Tensor(
[[[ 1.1676207  -0.2399907  -0.48948863 -0.43814147]
  [-0.11419237  0.06112379  0.4712959  -0.41822734]
  [ 0.6345568  -0.9779921   0.04900781  0.2944275 ]
  [ 1.2179315  -0.09482005 -0.9809958  -0.14211564]]], shape=(1, 4, 4), dtype=float32)
"""
```
```
Outputs:
tf.Tensor(
[[[ 1.1676207  -0.2399907  -0.48948863 -0.43814147]
  [-0.11419237  0.06112379  0.4712959  -0.41822734]
  [ 0.6345568  -0.9779921   0.04900781  0.2944275 ]
  [ 1.2179315  -0.09482005 -0.9809958  -0.14211564]]], shape=(1, 4, 4), dtype=float32)
```

Now we are done building the embedding layer!

The encoder is composed of multiple encoder layers that are stacked together. Each encoder layer takes a sequence of embeddings as input and processes them through two sublayers: a `multi-head self-attention` layer and a `fully connected feed-forward` layer. The output embeddings from each encoder layer maintain the same size as the input embeddings. The primary purpose of the encoder stack is to modify the input embeddings in order to create representations that capture contextual information within the sequence in a hierarchical manner. For instance, if the words "keynote" or "phone" are in proximity to the word "apple," the encoder will adjust the embedding of "apple" to reflect more of a "company-like" context rather than a "fruit-like" one.

Each of these sublayers also uses skip connections and layer normalization, which are standard tricks to train deep neural networks effectively. But to truly understand what makes a transformer work, we have to go deeper. Let’s start with the most important building block: the *self-attention* layer.

الآن انتهينا من بناء طبقة التضمين!

يتكون المشفر من طبقات تشفير متعددة مكدسة معًا. تأخذ كل طبقة تشفير سلسلة من التضمينات كمدخلات وتعالجها من خلال طبقتين فرعيتين: طبقة *الانتباه الذاتي متعدد الرؤوس* وطبقة *التغذية الأمامية المتصلة بالكامل*. تحافظ التضمينات الناتجة من كل طبقة تشفير على نفس حجم التضمينات المدخلة. الغرض الأساسي من تكديس طبقات التشفير هو تعديل عمليات التضمين المُدخلة من أجل إنشاء تمثيلات تلتقط المعلومات السياقية داخل التسلسل بطريقة هرمية. على سبيل المثال، إذا كانت الكلمات "keynote" أو "phone" قريبة من كلمة "apple"، فسيقوم  المُشفّر بتعديل تضمين "apple" ليعكس سياق "يشبه الشركة" بدلاً من سياق "يشبه الفاكهة".

تستخدم كل طبقة من هذه الطبقات الفرعية أيضًا وصلات التخطي وتقييس الطبقة، وهي حيل قياسية لتدريب الشبكات العصبية العميقة بفعالية. ولكن لكي نفهم حقًا ما الذي يجعل المحولات تعمل، علينا أن نتعمق أكثر. لنبدأ بأهم لبنة: طبقة *الانتباه الذاتي*.

<br>
<br>

### Self-Attention

Self-attention, also known as intra-attention, is a mechanism in the Transformer architecture that allows an input sequence to attend to other positions within itself. It is a key component of both the encoder and decoder modules in Transformers. In self-attention, each position in the input sequence generates three vectors: Query (Q), Key (K), and Value (V). These vectors are linear projections of the input embeddings. The self-attention mechanism then computes a weighted sum of the values (V) based on the similarity between the query (Q) and key (K) vectors. The weights are determined by the dot product between the query and key vectors, followed by an application of the softmax function to obtain the attention distribution. This attention distribution represents the importance or relevance of each position to the current position.

The weighted sum of values, weighted by the attention distribution, is the output of the self-attention layer. This output captures the contextual representation of the input sequence by considering the relationships and dependencies between different positions. The self-attention mechanism allows each position to attend to all other positions, enabling the model to capture long-range dependencies and contextual information effectively. One common implementation of self-attention, known as **scaled dot-product attention**, is widely used and described in the Vaswani et al. paper. This approach involves several steps to calculate the attention scores and update the token embeddings:

1. The token embeddings are projected into three vectors: query, key, and value.
2. Attention scores are computed by measuring the similarity between the query and key vectors using the dot product. This is efficiently achieved through matrix multiplication of the embeddings. Higher dot product values indicate stronger relationships between the query and key vectors, while low values indicate less similarity. The resulting attention scores form an n × n matrix, where n represents the number of input tokens.
3. To ensure stability during training, the attention scores are scaled by a factor to normalize their variance. Then, a softmax function is applied to normalize the column values, ensuring they sum up to 1. This produces the attention weights, which also form an n × n matrix.
4. The token embeddings are updated by multiplying them with their corresponding attention weights and summing the results. This process generates an updated representation for each embedding, taking into account the importance assigned to each token by the attention mechanism.


الانتباه الذاتي، المعروف أيضًا باسم الانتباه الداخلي، هو آلية في بنية المحولات تسمح للوحدات النصية في سلسلة الإدخال -بالاطلاع على الوحدات النصية الأخرى التي تجاورها. إنّه مكون رئيسي لكل من وحدات التشفير وفك التشفير في المحولات. في الانتباه الذاتي، يكون لكل وحدة نصية في تسلسل الإدخال ثلاثة متجهات: الاستعلام (Q) والمفتاح (K) والقيمة (V). هذه المتجهات هي إسقاطات خطية للتضمينات المُدخلة. تحسب آلية الانتباه الذاتي بعد ذلك المجموع الموزون للقيم (V) بناءً على التشابه بين متجهات الاستعلام (Q) والمتجهات المفتاحية (K). يتم تحديد الأوزان بواسطة الجداء النقطي بين الاستعلام والمفتاح، متبوعًا بتطبيق دالة softmax للحصول على توزيع الانتباه. يمثل توزيع الانتباه هذا أهمية أو صلة كل وحدة نصية بالأخرى.

المجموع الموزون للقيم، الموزون من خلال قيم توزيع الانتباه، هو ناتج طبقة الانتباه الذاتي. يلتقط هذا الخرج التمثيل السياقي لتسلسل الإدخال من خلال النظر في العلاقات والتبعيات بين المواضع أو الوحدات النصية المختلفة. تتيح آلية الانتباه الذاتي لكل موضع برؤية جميع المواضع الأخرى، مما يمكّن النموذج من التقاط التبعيات بعيدة المدى والمعلومات السياقية بشكل فعال. يتم استخدام أحد التطبيقات الشائعة للانتباه الذاتي، والمعروف باسم **اانتباه الجداء النقطي الموزون**، على نطاق واسع وتم وصفه في ورقة فاسواني. يتضمن هذا النهج عدة خطوات لحساب درجات الانتباه وتحديث التضمينات:

1. يتم عمل ثلاث إسقاطات خطية لتضمينات الوحدات النصية للحصول على ثلاثة متجهات: الاستعلام والمفتاح والقيمة.
2. يتم حساب درجات الانتباه عن طريق قياس التشابه بين الاستعلام والمفتاح باستخدام الجداء النقطي. يتم تحقيق ذلك بكفاءة من خلال ضرب المصفوفات. تشير قيم الجداء النقطي الأعلى إلى علاقات أقوى بين الاستعلام والمفتاح، بينما تشير القيم المنخفضة إلى تشابه أقل. تشكل درجات الانتباه الناتجة مصفوفة n × n، حيث تمثل n عدد الوحدات النصية الفريدة الموجودة في الدخل.
3. لضمان الاستقرار أثناء التدريب، يتم تحجيم درجات الانتباه بعامل تقييس لتباينها. بعد ذلك، يتم تطبيق دالة softmax للحصول على توزيع الانتباه. ينتج عن ذلك أوزان الانتباه، والتي تشكل أيضًا مصفوفة n × n.
4. يتم تحديث التضمينات للوحدات النصية (تمثيلات الكلمات) بضربها مع أوزان الانتباه المقابلة لها وجمع النتائج. تنشئ هذه العملية تمثيلاً مُحدّثًا لكل تضمين، مع الأخذ في الاعتبار أهمية كل كلمة بالنسبة للأخرى من خلال آلية الانتباه.

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

    def call(self, query, key, value, mask=None):
        """
        Applies attention mechanism to the input query, key, and value tensors.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            mask: Optional mask tensor.

        Returns:
            Updated value embeddings after applying attention mechanism.
        """
        query = self.query_weights(query)
        key = self.key_weights(key)
        value = self.value_weights(value)

        att_scores = tf.matmul(query, tf.transpose(key, perm=[0, 2, 1])) / tf.math.sqrt(tf.cast(tf.shape(query)[-1], tf.float32))

        if mask is not None:
            mask = tf.cast(mask, dtype=tf.bool)
            att_scores = tf.where(mask, att_scores, tf.constant(-1e9, dtype=att_scores.dtype))

        att_weights = tf.nn.softmax(att_scores, axis=-1)
        n_value = tf.matmul(att_weights, value)

        return n_value

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

We’ve initialized three independent linear layers that apply matrix multiplication to the embedding vectors to produce tensors of shape *[batch_size, seq_len, head_dim]*, where `head_dim` is the number of dimensions we are projecting into. In practically, `head_dim` be smaller than the number of embedding dimensions of the tokens (`embed_dim`), in practice it is chosen to be a multiple of `embed_dim` so that the computation across each head is constant. For example, BERT has *12* attention heads, so the dimension of each head is *768/12=64*. There is a clear advantage to incorporating multiple sets of linear projections, each representing an attention head. But why is it necessary to have more than one attention head? The reason is that when using just a single head, the softmax tends to focus primarily on one aspect of similarity. Having multiple heads provides a form of redundancy. If one head fails to learn the correct attention pattern, other heads can still capture the relevant information. This redundancy can improve the robustness of the model and reduce overfitting. Another advantage is that attention heads operate independently, allowing for parallelization during training and inference, which speeds up computations.


لقد قمنا بتهيئة ثلاث طبقات خطية مستقلة لتطبيق عمليات الضرب المصفوفية على متجهات التضمين لإنتاج موترات من الشكل *[batch_size، seq_len، head_dim]* ، حيث `head_dim` هو عدد الأبعاد التي نُسقط إليها. في الممارسات العملية تكون `head_dim` أصغر من عدد أبعاد التضمين `embed_dim`، ويتم اختياره ليكون من مضاعفات `embed_dim` بحيث يكون حجم كل رأس ثابتًا. على سبيل المثال، يحتوي BERT على رؤوس انتباه *12*، وبالتالي فإن بُعد كل رأس هو *768/12=64*. من الواضح أن الدمج بين مجموعات متعددة من الإسقاطات الخطية، بحيث كل منها يُمثل رأس انتباه. لكن لماذا من الضروري أن يكون لديك أكثر من رأس انتباه؟ والسبب هو أنه عند استخدام رأس واحد فقط، يميل softmax إلى التركيز على جانب واحد من أوجه التشابه. كما أن وجود عدة رؤوس يوفر شكلاً من أشكال التكرار. إذا فشل رأس واحد في تعلم نمط الانتباه الصحيح، فلا يزال بإمكان الرؤوس الآخرى التقاط المعلومات ذات الصلة. يمكن أن يؤدي هذا التكرار إلى تحسين متانة النموذج وتقليل الضبط المُفرط. ميزة أخرى هي أن رؤوس الانتباه تعمل بشكل مستقل، مما يسمح بالتوازي أثناء التدريب والاستدلال، مما يؤدي إلى تسريع العمليات الحسابية.

<br>
<br>

### Multi-headed attention

By introducing multiple attention heads, the model gains the ability to simultaneously focus on multiple aspects. For instance, one head can attend to subject-verb interactions, while another head can identify nearby adjectives. This multi-head approach empowers the model to capture a broader range of semantic relationships within the sequence, enhancing its understanding and representation capabilities. Now that we have a single attention head, we can concatenate the outputs of each one to implement the full multi-head attention layer:

من خلال تقديم رؤوس انتباه متعددة، يكتسب النموذج القدرة على التركيز في وقت واحد على جوانب متعددة. على سبيل المثال، يمكن لرأس واحد أن يفهم تفاعلات الفاعل والفعل، بينما يمكن لرأس آخر تحديد الصفات المُجاورة. يُمكّن النهج متعدد الرؤوس النموذج من التقاط نطاق أوسع من العلاقات الدلالية ضمن التسلسل، مما يعزز قدراته في الفهم والتمثيل. الآن بعد أن أصبح لدينا رأس انتباه واحد، يمكننا وصل مخرجات عدة رؤوس لتحقيق طبقة الانتباه متعددة الرؤوس الكاملة (كل رأس انتباه يُعطي 64 بُعد، فيتم وصلها لإعادة تشكيل أبعاد التضمين الأصلية 768):

```
class MultiHead_Attention(tf.keras.layers.Layer):
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

    def call(self, query, key, value, mask=None):
        """
        Applies multi-head attention mechanism to the input query, key, and value tensors.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            mask: Optional mask tensor.

        Returns:
            Updated hidden state after applying multi-head attention mechanism.
        """
        attention_outputs = [attention_head(query, key, value, mask=mask) for attention_head in self.attention_heads]
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

Notice that the concatenated output from the attention heads is also fed through a final linear layer to produce an output tensor of shape *[batch_size, seq_len, hidden_dim]* that is suitable for the feed-forward network downstream.

لاحظ أن الخرج الموصول من رؤوس الانتباه يتم تمريره إلى طبقة خطية أخيرة لإنتاج موتر بالشكل *[batch_size, seq_len, hidden_dim]* مناسب لشبكة التغذية الأمامية النهائية.

Testing:
```
# Define the configuration
class Config:
    def __init__(self):
        self.sequence_length = 4
        self.hidden_size = 4
        self.frequency_factor = 10000
        self.max_position_embeddings = 4
        self.source_vocab_size = 10
        self.target_vocab_size = 10
        self.positional_information_type = 'embs'
        self.hidden_dropout_prob = 0.1
        self.num_heads = 2

config = Config()

Embeddings_layer = Embeddings(config, 10)

# Create a sample input tensor with token IDs
input_ids = tf.constant([[2, 2, 0, 0]])

# Apply Embeddings_layer
x = Embeddings_layer(input_ids)

# Apply MultiHeadAttention
multihead_attn = MultiHead_Attention(config)
x = multihead_attn(x, x, x)

print("Outputs:")
print(x)
```
```
Outputs:
tf.Tensor(
[[[-2.574672   -0.17047548  1.3618734  -0.37746006]
  [-2.5156658  -0.16493037  1.3059906  -0.39567095]
  [-2.4666185  -0.09850129  1.303617   -0.2874905 ]
  [-2.549143   -0.1712878   1.3354063  -0.39174497]]], shape=(1, 4, 4), dtype=float32)
```
<br>
<br>

### The Feed-Forward Layer and Normalization

The feed-forward sublayer in both the encoder and decoder modules can be described as a simple two-layer fully connected neural network. However, its operation differs from a standard network in that it treats each embedding in the sequence independently rather than processing the entire sequence as a single vector. Because of this characteristic, it is often referred to as a **position-wise feed-forward layer**. In the literature, a general guideline suggests setting the hidden size of the first layer to be four times the size of the embeddings. Additionally, a GELU activation function is commonly used in this layer. It is believed that this particular sublayer contributes significantly to the model's capacity and memorization abilities. Consequently, when scaling up the models, this layer is often a focal point for adjustment and expansion.

يمكن وصف الطبقة الفرعية للتغذية الأمامية في كل من وحدات التشفير وفك التشفير على أنها شبكة عصبية بسيطة من طبقتين متصلتين بالكامل. ومع ذلك، فإن عملها يختلف عن الشبكة القياسية من حيث أنه يتعامل مع كل تضمين في التسلسل بشكل مستقل بدلاً من معالجة التسلسل بأكمله كمتجه واحد. بسبب هذه الخاصية، يُشار إليها غالبًا باسم ** طبقة تغذية أمامية موضع بموضع**. في الأدبيات (المراجع العلمية)، يقترح النهج العام تحديد حجم للطبقة الأولى ليكون أربعة أضعاف حجم التضمينات. بالإضافة إلى ذلك، يتم استخدام دالة تنشيط GELU بشكل شائع في هذه الطبقة. من المُعتقد أن هذه الطبقة الفرعية تساهم بشكل كبير في سعة استيعاب النموذج للمعلومات وقدراته على الحفظ. وبالتالي، عند توسيع نطاق النماذج، غالبًا ما تكون هذه الطبقة نقطة محورية للضبط والتوسيع.


    
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

**Adding Layer Normalization.** When it comes to placing layer normalization in the encoder or decoder layers of a transformer, there are two main choices that have been widely adopted in the literature. The first choice is to apply layer normalization before each sub-layer, which includes the self-attention and feed-forward sub-layers. This means that the input to each sub-layer is normalized independently , it's called **Pre layer normalization**. The second choice is to apply layer normalization after each sub-layer, which means that the normalization is applied to the output of each sub-layer, it's called **Post layer normalization**. Both approaches have their own advantages and have been shown to be effective in different transformer architectures. The choice of placement often depends on the specific task and architecture being used.
<br>
<br>

### Encoder layer

Now that we've built all the main parts of the encoder layer, we'll put them together to build it:

```
class Encoder(tf.keras.layers.Layer):
    """
    Encoder layer implementation.

    Args:
        config: Configuration object.

    Attributes:
        multihead_attention: Multi-head attention layer.
        norm1: Layer normalization layer.
        norm2: Layer normalization layer.
        feed_forward: Feed-forward layer.
        dropout: Dropout layer.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.multihead_attention = MultiHead_Attention(config)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.feed_forward = FeedForward(config)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_state, mask=None, training=False):
        """
        Applies the encoder layer to the input hidden state.

        Args:
            hidden_state: Hidden state tensor (bs, len, dim).
            mask: Padding mask tensor (bs, len, len) or (bs, 1, len) or None.
            training: Boolean flag indicating whether the layer is in training mode or not.

        Returns:
            Updated hidden state after applying the encoder layer.
        """

        attention_output = self.multihead_attention(hidden_state, hidden_state, hidden_state, mask = None)  # Apply multi-head attention
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

Testing:

```
# Define the configuration
class Config:
    def __init__(self):
        self.sequence_length = 4
        self.hidden_size = 4
        self.frequency_factor = 10000
        self.max_position_embeddings = 4
        self.source_vocab_size = 10
        self.target_vocab_size = 10
        self.positional_information_type = 'embs'
        self.hidden_dropout_prob = 0.1
        self.num_heads = 2
        self.intermediate_fc_size = self.hidden_size * 4.


config = Config()

Embeddings_layer = Embeddings(config, 10)

# Create a sample input tensor with token IDs
batch_size = 2
seq_length = 4
input_ids = tf.random.uniform((batch_size, seq_length), maxval=config.sequence_length, dtype=tf.int32)

# Apply Embeddings_layer
x = Embeddings_layer(input_ids)

# Apply Encoder
encoder = Encoder(config)
x = encoder(x)

print("Outputs:")
print(x)
```
```
Outputs:
tf.Tensor(
[[[-0.28684267  0.588221    1.1763539  -1.4777321 ]
  [-0.5236694  -0.9702377   1.6593678  -0.1654609 ]
  [-0.5649392  -1.3195349   1.225875    0.6585991 ]
  [ 0.5740138   0.3637715  -1.712724    0.77493864]]

 [[-0.55802745 -0.29481202  1.6977357  -0.8448962 ]
  [-0.4659584  -1.4050406   1.1019287   0.7690702 ]
  [-0.34269786  0.55238974  1.2276003  -1.4372921 ]
  [ 0.02217576  0.5394346  -1.6122792   1.0506687 ]]], shape=(2, 4, 4), dtype=float32)
```

We’ve now implemented our first transformer encoder layer from scratch!

<br>
<br>

### Decoder

The main difference between the decoder and encoder is that the decoder has two attention sublayers:

1. **Masked multi-head self-attention layer.** Ensures that the tokens we generate at each timestep are only based on the past outputs and the current token being predicted. Without this, the decoder could cheat during training by simply copying the target translations; masking the inputs ensures the task is not trivial.
2. **Encoder-decoder attention layer.** Performs multi-head attention over the output key and value vectors of the encoder stack, with the intermediate representations of the decoder acting as the queries. This way the encoder-decoder attention layer learns how to relate tokens from two different sequences, such as two different languages. The decoder has access to the encoder keys and values in each block.

Let’s take a look at the modifications we need to make to include masking in our self-attention layer. The trick with masked self-attention is to introduce a mask matrix with ones on the lower diagonal and zeros above:

```
import tensorflow as tf

seq_len = 4
mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
mask = tf.expand_dims(mask, axis=0)
mask
"""
<tf.Tensor: shape=(1, 4, 4), dtype=float32, numpy=
array([[[1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.]]], dtype=float32)>
"""
```

Here we've used TensorFlow's `tf.linalg.band_part()` function to create the lower triangular matrix. Once we have this mask matrix, we can prevent each attention head from peeking at future tokens by using `tf.where()` to replace all the zeros with negative infinity:

```
import tensorflow as tf

# Create example query, key, and value tensors
scores = tf.random.normal(shape=(1, 4, 4))

# Apply the mask to scores tensor
scores = tf.where(tf.equal(mask, 0), tf.constant(-float("inf"), dtype=tf.float32), scores) # or `scores += (1 - tf.cast(mask, dtype=tf.float32)) * -1e9`

# Print the scores tensor
print(scores)
"""
tf.Tensor(
[[[ 0.78525937        -inf        -inf        -inf]
  [-0.43159866 -0.37508744        -inf        -inf]
  [ 0.7587768  -0.1002408  -1.6593473         -inf]
  [ 0.25996745 -1.0069757   1.1573174  -1.1290911 ]]], shape=(1, 4, 4), dtype=float32)
"""
```

By setting the upper values to negative infinity, we guarantee that the attention weights are all zero once we take the softmax over the scores because e^(-∞) = 0 (recall that softmax calculates the normalized exponential).

```
class Decoder(tf.keras.layers.Layer):
    """
    Decoder layer implementation.

    Args:
        config: Configuration object containing hyperparameters.

    Attributes:
        supports_masking: Boolean indicating if the layer supports masking.
        masked_multihead_attention: Masked multi-head attention layer.
        multihead_attention: Multi-head attention layer.
        norm1: Layer normalization layer.
        norm2: Layer normalization layer.
        norm3: Layer normalization layer.
        feed_forward: Feed-forward layer.
        dropout: Dropout layer.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.masked_multihead_attention = MultiHead_Attention(config)
        self.multihead_attention = MultiHead_Attention(config)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.norm3 = tf.keras.layers.LayerNormalization()
        self.feed_forward = FeedForward(config)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_state, encoder_info, mask=None, training=False):
        """
        Applies the decoder layer to the input hidden state.

        Args:
            hidden_state: Hidden state tensor.
            encoder_info: Encoder information tensor.
            mask: Optional mask tensor.
            training: Boolean indicating if the model is in training mode.

        Returns:
            Updated hidden state after applying the decoder layer.
        """
        input_shape = tf.shape(hidden_state)
        causal_mask = self.get_causal_attention_mask(hidden_state)

        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output = self.masked_multihead_attention(hidden_state, hidden_state, hidden_state, mask=causal_mask)
        hidden_state = self.norm1(attention_output + hidden_state)
        attention_output = self.multihead_attention(hidden_state, encoder_info, encoder_info, mask=padding_mask)
        hidden_state = self.norm2(attention_output + hidden_state)
        feed_forward_output = self.feed_forward(hidden_state)
        hidden_state = self.norm3(feed_forward_output + hidden_state)
        hidden_state = self.dropout(hidden_state, training=training)
        return hidden_state

    def get_config(self):
        """
        Returns the configuration of the decoder layer.

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

    def get_causal_attention_mask(self, inputs):
        """
        Generates the causal attention mask.

        Args:
            inputs: Input tensor.

        Returns:
            Causal attention mask tensor.
        """
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype=tf.int32)
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)
```

Testing:

```
class Config:
    def __init__(self):
        self.sequence_length = 4
        self.hidden_size = 4
        self.frequency_factor = 10000
        self.max_position_embeddings = 4
        self.source_vocab_size = 10
        self.target_vocab_size = 10
        self.positional_information_type = 'embs'
        self.hidden_dropout_prob = 0.1
        self.num_heads = 2
        self.intermediate_fc_size = self.hidden_size * 4

config = Config()

embeddings_layer1 = Embeddings(config, 10)
embeddings_layer2 = Embeddings(config, 10)

batch_size = 2
seq_length = 4
input_ids1 = tf.constant([[2, 1, 3, 0], [2, 1, 3, 0]])
input_ids2 = tf.constant([[1, 3, 0, 0], [2, 1, 3, 0]])

x1 = embeddings_layer1(input_ids1)
x2 = embeddings_layer2(input_ids2)

encoder = Encoder(config)
decoder = Decoder(config)

enc_out = encoder(x1)
enc_out = tf.keras.layers.Masking()(enc_out)

x = decoder(x2, enc_out)

print("Outputs:")
print(x)
```

```
Outputs:
tf.Tensor(
[[[-1.6775297   0.87867206  0.6106412   0.18821625]
  [-0.7511008  -1.0770721   0.36531034  1.4628625 ]
  [-0.6215651  -0.9170716   1.6596491  -0.12101232]
  [-0.46419287 -1.323965    0.430033    1.3581249 ]]

 [[-0.867159    1.1805613   0.7913892  -1.1047916 ]
  [-1.4458524  -0.21441776  0.33897528  1.3212948 ]
  [-0.6998376  -0.6339283  -0.38511607  1.7188822 ]
  [-0.823983   -0.6761114  -0.18121609  1.6813104 ]]], shape=(2, 4, 4), dtype=float32)
```

Now we have finished building the main components of the model!

<br>
<br>

### Transformer Model

Now, after we have built the necessary ingredients, and tested them. We can build in safety our transformer model.

```
import tensorflow as tf

class Transformer(tf.keras.Model):
    """
    Transformer model implementation for sequence-to-sequence tasks.

    Args:
        config: Configuration object containing model hyperparameters.
        source_vocab_size: The vocabulary size of the source language.
        target_vocab_size: The vocabulary size of the target language.

    Attributes:
        enc_embed_layer: Embeddings layer for the encoder inputs.
        dec_embed_layer: Embeddings layer for the decoder inputs.
        encoder: List of encoder layers.
        decoder: List of decoder layers.
        dropout: Dropout layer for regularization.
        output_layer: Dense layer for output prediction.

    Methods:
        call: Forward pass of the transformer model.
        get_config: Returns the configuration dictionary of the transformer model.
    """

    def __init__(self, config, source_vocab_size, target_vocab_size):
        super(Transformer, self).__init__()
        self.enc_embed_layer = Embeddings(config, source_vocab_size)
        self.dec_embed_layer = Embeddings(config, target_vocab_size)
        self.encoder = [Encoder(config) for _ in range(config.num_layers)]
        self.decoder = [Decoder(config) for _ in range(config.num_layers)]
        self.dropout = tf.keras.layers.Dropout(config.final_dropout_prob)
        self.output_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training=False):
        """
        Forward pass of the transformer model.

        Args:
            inputs: Input data.
            training: Boolean flag indicating whether the model is in training mode or not.

        Returns:
            Output logits of the transformer model.
        """
        source_inputs = inputs["encoder_inputs"]
        target_inputs = inputs["decoder_inputs"]

        x_enc = self.enc_embed_layer(source_inputs)
        x_dec = self.dec_embed_layer(target_inputs)

        for encoder_layer in self.encoder:
            x_enc = encoder_layer(x_enc, training=training)

        # Remove the mask used in the encoder as it's not needed in the decoder
        x_enc._keras_mask = None

        for decoder_layer in self.decoder:
            x_dec = decoder_layer(x_dec, x_enc, training=training)

        x_dec = self.dropout(x_dec, training=training)
        x_logits = self.output_layer(x_dec)

        # Remove the mask from the logits as it's not needed in the loss function
        x_logits._keras_mask = None

        return x_logits

    def get_config(self):
        """
        Returns the configuration dictionary of the transformer model.

        Returns:
            Configuration dictionary.
        """
        config = super(Transformer, self).get_config()
        # Add custom configurations to the dictionary if needed
        return config
```

All are done, Let's go train our model!

<br>
<br>

## End-to-end Transformer

<br>
<br>
We will train the end-to-end Transformer model, which is responsible for mapping the source sequence and the target sequence to predict the target sequence one step ahead. This model seamlessly integrates the components we have developed: the Embedding layer, the Encoder, and the Decoder. Both the Encoder and the Decoder can be stacked to create more powerful versions as they maintain the same shape.
<br>
<br>

### The Schedular

Before training the Transformer, we need to determine the training strategy. In accordance with the paper *Attention Is All You Need*, we will utilize the Adam optimizer with a custom learning rate schedule. One technique we will employ is known as learning rate warmup. This technique gradually increases the learning rate during the initial iterations of training in order to enhance stability and accelerate convergence.

During the warmup phase, the learning rate is increased in a linear manner or according to a specific schedule until it reaches a predefined value. The objective of this warmup phase is to enable the model to explore a wider range of solutions during the initial training phase when the gradients might be large and unstable. The specific formula for the learning rate warmup is as follows:

```
learning_rate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
```

Here, `d_model` represents the dimensionality of the model, `step_num` indicates the current training step, and `warmup_steps` denotes the number of warmup steps. Typically, `warmup_steps` is set to a few thousand or a fraction of the total training steps. The motivation behind learning rate warmup is to address two challenges that often occur at the beginning of training:

1. **Large Gradient Magnitudes.** In the initial stages of training, the model parameters are randomly initialized, and the gradients can be large. If a high learning rate is applied immediately, it can cause unstable updates and prevent the model from converging. Warmup allows the model to stabilize by starting with a lower learning rate and gradually increasing it.
2. **Exploration of the Solution Space.** The model needs to explore a wide range of solutions to find an optimal or near-optimal solution. A high learning rate at the beginning may cause the model to converge prematurely to suboptimal solutions. Warmup enables the model to explore the solution space more effectively by starting with a low learning rate and then increasing it to search for better solutions.

Let's implement the scheduler:

```
import tensorflow as tf

class LrSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate schedule with warmup for training.

    Args:
        config: Configuration object containing hyperparameters.

    Attributes:
        warmup_steps: Number of warmup steps for learning rate warmup.
        d: Hidden size of the model.
    """

    def __init__(self, config):
        super().__init__()
        self.warmup_steps = config.warmup_steps
        self.d = tf.cast(config.hidden_size, tf.float32)

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
        lr =  tf.math.rsqrt(self.d) * tf.math.minimum(arg1, arg2)
        return lr

    def get_config(self):
        """
        Returns the configuration of the custom learning rate schedule.

        Returns:
            Configuration dictionary.

        """
        config = {
            "warmup_steps": self.warmup_steps,
        }
        return config
```

This corresponds to increasing the learning rate linearly for the first `warmup_steps` training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. You can see how the learning rate values change with each time step with the following code:

```
import matplotlib.pyplot as plt

class Config:
    def __init__(self):
        self.sequence_length = 4
        self.hidden_size = 4
        self.frequency_factor = 10000
        self.max_position_embeddings = 4
        self.source_vocab_size = 10
        self.target_vocab_size = 10
        self.positional_information_type = 'embs'
        self.hidden_dropout_prob = 0.1
        self.num_heads = 2
        self.intermediate_fc_size = self.hidden_size * 4
        self.warmup_steps = 4000


config = Config()

d = 768
lr = LrSchedule(config)
optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

plt.plot(lr(tf.range(40000, dtype=tf.float32)))
plt.ylabel('Learning Rate')
plt.xlabel('Train Step')
plt.show()
```

By gradually increasing the learning rate during the warmup phase, the model can effectively explore the search space, adapt better to the training data, and ultimately converge to a more optimal solution. Once the warmup phase is completed, the learning rate follows its regular schedule, which may involve decay or a fixed rate, for the remaining training iterations.

### Loss Function

Next, we are required to specify the loss function for the training process. In this particular model, an additional step is needed where a mask is applied to the output. This mask ensures that the loss (and accuracy) calculations are performed only on the non-padding elements, disregarding any padded values:

```
import tensorflow as tf

def scce_masked_loss(label, pred):
    """
    Computes the masked Sparse Categorical Cross Entropy (SCCE) loss between the predicted and target labels.

    Args:
        label: Target label tensor.
        pred: Predicted logit tensor.

    Returns:
        Masked loss value.
    """
    # Create a mask to ignore padded tokens
    mask = label != 0

    # Use Sparse Categorical Cross Entropy loss with no reduction
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    # Compute the loss without reducing, which will return a loss value for each token
    loss = loss_object(label, pred)

    # Apply the mask to ignore padded tokens in the loss calculation
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    # Compute the average loss over non-padded tokens
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss
```


When we are getting talking about loss function, we must refer to the *Label Smoothing*. It's a regularization technique commonly used in deep learning models, particularly in classification tasks, to improve generalization and prevent the model from becoming overly confident in its predictions. It addresses the issue where a model may assign a probability close to 1 to the predicted class and 0 to all other classes, leading to overfitting and potential sensitivity to small perturbations in the input. Label smoothing introduces a small amount of uncertainty or noise into the training process by modifying the target (ground truth) labels. Instead of using one-hot encoded labels with a single element set to 1 and all others set to 0, label smoothing distributes the probability mass among multiple classes ([read more](https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06)). In empirical studies, the improvement in machine translation performance due to label smoothing is typically in the range of 0.5% to 2%. However, these numbers are not fixed and can vary based on different factors. For certain datasets or model architectures, the improvement might be more significant, while for others, it might be less noticeable. One of the main challenges with using one-hot encoding and label smoothing in machine translation tasks with large vocabularies is the high dimensionality of the one-hot encoded vectors. As the vocabulary size increases, the one-hot encoded vectors become very sparse, leading to memory and computational inefficiencies. Here, we are not going to use it because of the limitations in the sources that we have. However, you could try it easily just by replacing `SparseCategoricalCrossentropy` with [`CategoricalCrossentropy`](https://github.com/keras-team/keras/tree/v2.13.1//keras/losses.py#L837), using the `label_smoothing` parameter, and representing the target sentences with one-hot encoding. Here's how you can use technique:


```
import tensorflow as tf

def cce_loss(label, pred):
    """
    Computes the Categorical Cross Entropy (CCE) loss with optional label smoothing.

    Args:
        label: Target label tensor.
        pred: Predicted logit tensor.

    Returns:
        Computed CCE loss value.
    """
    # Create a mask to ignore padded tokens
    mask = label != 0

    # Use Categorical Cross Entropy with optional label smoothing
    scc_loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=0.1, reduction='none')

    # Convert label to one-hot encoding
    label = tf.one_hot(tf.cast(label, tf.int32), config.target_vocab_size)

    # Compute the loss with the label smoothing
    loss = scc_loss(label, pred)

    # Apply the mask to ignore padded tokens in the loss calculation
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    # Compute the average loss over non-padded tokens
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss
```

### Metrics

It's common to use masked accuracy with this task, so let's implement it:

```
import tensorflow as tf

def masked_accuracy(label, pred):
    """
    Computes the masked accuracy between the predicted and target labels.

    Args:
        label: Target label tensor.
        pred: Predicted label tensor.

    Returns:
        Masked accuracy value.
    """
    # Get the predicted labels by taking the argmax along the last dimension
    pred_labels = tf.argmax(pred, axis=2)

    # Convert the target labels to the same data type as the predicted labels
    label = tf.cast(label, pred_labels.dtype)

    # Compute a binary tensor for matching predicted and target labels
    match = label == pred_labels

    # Create a mask to ignore padded tokens
    mask = label != 0

    # Apply the mask to the matching tensor
    match = match & mask

    # Convert the binary tensor to floating-point values
    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    # Compute the accuracy over non-padded tokens
    return tf.reduce_sum(match) / tf.reduce_sum(mask)
```

However, it's important to note that accuracy alone may not provide a complete picture of translation quality. Translation evaluation often requires the use of specialized metrics like *BLEU*, *METEOR*, *ROUGE*, or *CIDEr*, which consider the quality, fluency, and semantic similarity of the translations compared to reference translations. These metrics take into account various aspects of translation such as word choice, word order, and overall coherence. Therefore, while the `masked_accuracy` function can be used as a basic measure of accuracy, it is advisable to complement it with established translation evaluation metrics for a more comprehensive assessment of translation quality.
<br>
<br>

BLEU (Bilingual Evaluation Understudy) is another metric used to evaluate the quality of machine translation output by comparing it to one or more reference translations. It was proposed as an automatic evaluation metric for machine translation systems and is widely used in the natural language processing (NLP) field.

The BLEU metric works by comparing the n-grams (contiguous sequences of n words) of the candidate translation (output) to the n-grams of the reference translations (ground truth). It calculates a precision score for each n-gram and then combines the scores using a modified geometric mean, giving more weight to shorter n-grams. BLEU ranges from 0 to 1, with 1 being a perfect match with the reference translations ([read more](https://machinelearningmastery.com/calculate-bleu-score-for-text-python/)).

Here's how the BLEU metric is calculated:

1. Calculate the n-gram precision for each n-gram size (usually up to 4-grams):
   - Count the number of n-grams in the candidate translation that appear in the reference translations.
   - Count the total number of n-grams in the candidate translation.

2. Calculate the brevity penalty to account for translations that are shorter than the references. This penalizes overly short translations that can achieve high precision but may not cover the full meaning of the input.

3. Combine the n-gram precisions using a modified geometric mean. The modified version addresses the issue of penalizing translations that have zero precision for a certain n-gram size. It replaces zero precisions with the lowest non-zero precision.

4. Multiply the combined n-gram precisions by the brevity penalty to get the final BLEU score.

```
import tensorflow as tf
import numpy as np
from collections import Counter

def compute_precision(candidate_ngrams, reference_ngrams):
    """
    Compute the precision of candidate n-grams with respect to reference n-grams.

    Args:
        candidate_ngrams: List of tuples representing candidate n-grams.
        reference_ngrams: List of tuples representing reference n-grams.

    Returns:
        Precision value.
    """
    candidate_counter = Counter(candidate_ngrams)
    reference_counter = Counter(reference_ngrams)

    # Calculate the intersection of n-grams in candidate and reference sentences
    intersection = sum((candidate_counter & reference_counter).values())

    # Total candidate n-grams
    total_candidate = sum(candidate_counter.values())

    # To avoid division by zero, set precision to a small value if there are no candidate n-grams
    precision = intersection / total_candidate if total_candidate > 0 else 1e-10

    return precision

def compute_bleu_batch(references_batch, candidates_batch, max_n=4):
    """
    Compute the masked BLEU score for a batch of sentences.

    Args:
        label: Target label tensor.
        pred: Predicted tensor.
        max_n: Maximum n-gram for BLEU computation.

    Returns:
        Computed masked BLEU score.
    """

    batch_size = len(references_batch)
    total_bleu_score = 0.0

    # Tokenize and compute n-grams for each candidate-reference pair in the batch
    for i in range(batch_size):
        references = references_batch[i]
        candidates = candidates_batch[i]

        precisions = []

        for candidate, reference in zip(candidates, references):
            candidate_tokens = candidate.split()
            reference_tokens = reference.split()

            # Calculate BLEU score for each n-gram up to max_n
            for n in range(1, max_n + 1):
                candidate_ngrams = [tuple(candidate_tokens[j:j + n]) for j in range(len(candidate_tokens) - n + 1)]
                reference_ngrams = [tuple(reference_tokens[j:j + n]) for j in range(len(reference_tokens) - n + 1)]

                precision_n = compute_precision(candidate_ngrams, reference_ngrams)
                precisions.append(precision_n)

        # Calculate the geometric mean of all the n-gram precisions for this candidate-reference pair
        geometric_mean = np.exp(np.mean(np.log(np.maximum(precisions, 1e-10))))

        # Calculate the brevity penalty for this candidate-reference pair
        reference_length = [len(reference.split()) for reference in references]
        candidate_length = [len(candidate.split()) for candidate in candidates]

        closest_refs = [min(reference_length, key=lambda x: abs(x - candidate_len)) for candidate_len in candidate_length]
        brevity_penalty = np.minimum(np.exp(1 - np.array(closest_refs) / np.array(candidate_length)), 1.0)

        # Calculate the BLEU score for this candidate-reference pair
        bleu_score = geometric_mean * brevity_penalty

        total_bleu_score += bleu_score

    # Calculate the average BLEU score over the entire batch
    average_bleu_score = total_bleu_score / batch_size

    return average_bleu_score

# Example usage with batch of sentences
references_batch = [["the quick brown fox jumped over the lazy dog"]]
candidates_batch = [["the quick brown fox jumped over the lazy dog from space"]]
bleu_score_batch = compute_bleu_batch(references_batch, candidates_batch)
print("Average BLEU Score:", bleu_score_batch)  # Output: 0.78
```

However, it also has some limitations, such as sensitivity to sentence length and the fact that it relies solely on n-gram matching without considering semantic meaning. As a result, researchers often use multiple evaluation metrics, including BLEU, to get a more comprehensive understanding of the translation system's performance.

**Note:** This implementation you can use it to test the performance of the model later, but you can't use it during training, it needs somewhat boring modifications in order to comply with Tensorflow's operations. So if you want to use it, it is better to experience the implementation of the [Keras_nlp](https://keras.io/api/keras_nlp/metrics/bleu/) package.

<br>
<br>

### Callbacks

```
import tensorflow as tf

class TransformerCallbacks(tf.keras.callbacks.Callback):
    """
    Custom callback to monitor the validation loss during training and save the best model.

    Args:
        config: Configuration object containing hyperparameters.

    Attributes:
        checkpoint_filepath: Filepath to save the best model.
        patience: Number of epochs to wait for improvement in validation loss.
        best_loss: Best validation loss observed during training.

    Methods:
        on_epoch_end: Called at the end of each epoch to monitor the validation loss.

    """

    def __init__(self, config):
        super(TransformerCallbacks, self).__init__()
        self.checkpoint_filepath = config.checkpoint_filepath
        self.patience = config.patience
        self.best_loss = float('inf')  # Initialize with a very large value for the first comparison

    def on_epoch_end(self, epoch, logs={}):
        """
        Callback function called at the end of each epoch to monitor the validation loss.

        Args:
            epoch: The current epoch number.
            logs: Dictionary containing training and validation metrics.

        """
        # Access the validation loss from the logs dictionary
        val_loss = logs.get('val_loss')

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.model.save_weights(self.checkpoint_filepath+'/'+'eng-fra-transformer.h5')
            print('The best model has been saved at epoch #{}'.format(epoch))
        elif self.patience:
            self.patience -= 1
            if self.patience == 0:
                self.model.stop_training = True
                print('Training stopped. No improvement after {} epochs.'.format(epoch))
```

### Transformer Configuration

```
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
            checkpoint_filepath: Filepath for saving model checkpoints.
            patience: Number of epochs with no improvement after which training will be stopped.

        """
        self.sequence_length = 60
        self.hidden_size = 256
        self.frequency_factor = 10000
        self.source_vocab_size = 16721
        self.target_vocab_size = 31405
        self.positional_information_type = 'embs'
        self.hidden_dropout_prob = 0.1
        self.num_heads = 8
        self.intermediate_fc_size = self.hidden_size * 4
        self.warmup_steps = 4000
        self.num_blocks = 2
        self.final_dropout_prob = 0.5
        self.epochs = 30
        self.checkpoint_filepath = '/content/drive/MyDrive/Colab Notebooks/NMT/tmp/checkpoint'
        self.patience = 3
```

### Monitor

```
import tensorflow as tf

config = Config()

# Create the Transformer model
transformer = Transformer(config, config.source_vocab_size, config.target_vocab_size)

# Create the learning rate schedule
lr = LrSchedule(config)

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
```

## Inference

## Conclusion
