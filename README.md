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




