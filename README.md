# Neural-Machine-Translator with Transformers
The task of translating text from one language into another is a common task. In this repo I'm going to construct and demonstrate a naive model for an English to French translation task using TensorFlow, which is inspired by the model presented in the "Attention Is All You Need" paper.

تعتبر مهمة ترجمة النص من لغة إلى لغة أخرى من المهام الشائعة. سأقوم في هذا الريبو ببناء وشرح نموذج بسيط لمهمة الترجمة من اللغة الإنجليزية إلى اللغة  الفرنسية باستخدام تنسرفلو، وهو مستلهم من النموذج الذي تم تقديمه في ورقة "الانتباه هو كل ماتحتاجه".

## Transformer architecture
The Transformer architecture is a popular model architecture used for various natural language processing tasks, including machine translation, text generation, and language understanding. It was introduced by Vaswani et al. in the paper "Attention Is All You Need."

The Transformer architecture consists of two main components: the encoder and the decoder. Both the encoder and decoder are composed of multiple layers of self-attention and feed-forward neural networks. The key idea behind the Transformer is the use of self-attention mechanisms, which allow the model to focus on different parts of the input sequence when generating the output:

المحولات هي بنية نموذجية شائعة تستخدم في العديد من مهام معالجة اللغة الطبيعية، بما في ذلك الترجمة الآلية وتوليد النص وفهم اللغة. تم تقديمه بواسطة Vaswani et al. في مقالة "الاهتمام هو كل ما تحتاجه".

تتكون بنية المحول من مكونين رئيسيين: المشفر ووحدة فك التشفير. يتكون كل من المشفر ومفكك التشفير من طبقات متعددة من الاهتمام الذاتي والشبكات العصبية ذات التغذية الأمامية. الفكرة الرئيسية وراء المحول هي استخدام آليات الانتباه الذاتي، والتي تسمح للنموذج بالتركيز على أجزاء مختلفة من سلسلة الدخل عند إنشاء سلسلة الخرج:


![Transformer architecture](imgs/transformer_arch.png "Transformer architecture")


### Here is a high-level overview of the Transformer architecture

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
import tensorflow as tf
text_file = tf.keras.utils.get_file(
    fname="fra-eng.zip",
    origin="http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip",
    extract=True,
)
# File location
text_file = pathlib.Path(text_file).parent / "fra.txt"
```
The dataset we're working on consists of 167,130 lines. Each line consists of the original sequence (the sentence in English) and the target sequence (in French).

### Parsing the data
We prepend the token "[start]" and we append the token "[end]" to the French sentence.

```
import pathlib
import pickle
import random
import re
import unicodedata

def normalize(line):
    """Normalize a line of text and split into two at the tab character"""
    line = unicodedata.normalize("NFKC", line.strip())
    line = re.sub(r"^([^ \w])(?!\s)", r"\1 ", line)
    line = re.sub(r"(\s[^ \w])(?!\s)", r"\1 ", line)
    line = re.sub(r"(?!\s)([^ \w])$", r" \1", line)
    line = re.sub(r"(?!\s)([^ \w]\s)", r" \1", line)
    eng, fra = line.split("\t")
    fra = "[start] " + fra + " [end]"
    return eng, fra

# normalize each line and separate into English and French
with open(text_file) as fp:
    text_pairs = [normalize(line) for line in fp]
with open("text_pairs.pickle", "wb") as fp:
    pickle.dump(text_pairs, fp)
```

Let's look at the data:
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

We need to write a function that associates each token with a unique integer number representing it to get what is called a "Tokens IDs". Fortunately, there is a layer in TensorFlow called [`TextVectorization`](https://keras.io/api/layers/preprocessing_layers/core_preprocessing_layers/text_vectorization/) that makes life easier for us. We'll use two instances of the TextVectorization layer to vectorize the text data (one for English and one for Spanish). 

