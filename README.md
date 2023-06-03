# Neural-Machine-Translator
The task of translating text from one language into another is a common task. In this repo I'm going to construct and demonstrate a naive model for an English to French translation task using TensorFlow, which is inspired by the model presented in the.

تعتبر مهمة ترجمة النص من لغة إلى لغة أخرى من المهام الشائعة. سأقوم في هذا الريبو ببناء وشرح نموذج ساذج لمهمة الترجمة من اللغة الإنجليزية إلى اللغة الفرنسية باستخدام تنسرفلو، وهو مستلهم من النموذج الذي تم تقديمه في ورقة .

## Transformer architecture
The Transformer architecture is a popular model architecture used for various natural language processing tasks, including machine translation, text generation, and language understanding. It was introduced by Vaswani et al. in the paper "Attention Is All You Need."

The Transformer architecture consists of two main components: the encoder and the decoder. Both the encoder and decoder are composed of multiple layers of self-attention and feed-forward neural networks. The key idea behind the Transformer is the use of self-attention mechanisms, which allow the model to focus on different parts of the input sequence when generating the output:

![Transformer Arc](imgs/transformer_arch.png)
