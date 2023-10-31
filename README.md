# English to Hawaiian Pidgin Translation | flan-t5-base-eng-hwp

## Access the model

 - **[Model:](https://huggingface.co/claudiatang/flan-t5-base-eng-hwp)** Hugging Face model page
 - **[Demo:](https://huggingface.co/spaces/claudiatang/English-Hawaiian-Pidgin-Translator)** Hugging Face Space 

## Introduction

This model is a fine-tuned version of [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) on a English and Hawaiian Pidgin dataset.
It achieves the following results on the evaluation set:
- Loss: 1.5058
- Bleu: 4.9532
- Gen Len: 18.8709

## Model description

### Running the model

The [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) documentation has more details on running the model.

However, to use this model to translate English to Hawaiian Pidgin, enter ``"translate English to Hawaiian Pidgin: "`` before your statement. 

For example, if you would like to translate "I went to Ala Moana today to go shopping" please tokenize all of the following: ``"translate English to Hawaiian Pidgin: I went to Ala Moana today to go shopping."``

If you are trying the [English-Hawaiian Pidgin Translator](https://huggingface.co/spaces/claudiatang/english_to_hawaiian-pidgin) space, there is no need for the input prefix, as it is automatically added.

## Training and evaluation data

There are not many English-Hawaiian Pidgin parallel corpora that are easily accessible. A parallel dataset, similar to [bible_para](https://huggingface.co/datasets/bible_para), was compiled by scraping the Hawaiʻi Pidgin Version (HWP) and the King James Version (KJV) from [biblegateway.com](https://www.biblegateway.com/). <!--- For more information, please refer to [this notebook](). -->

## Intended uses & limitations

Due to a limited set of training and evaluation data, this model has many limitations, such as not knowing certain Hawaiian Pidgin phrases or having trouble with longer sentences.

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0003
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 12

### Training results

| Training Loss | Epoch | Step | Validation Loss | Bleu   | Gen Len |
|:-------------:|:-----:|:----:|:---------------:|:------:|:-------:|
| No log        | 1.0   | 420  | 1.6158          | 3.6321 | 18.892  |
| 2.1318        | 2.0   | 840  | 1.4711          | 4.19   | 18.8743 |
| 1.5146        | 3.0   | 1260 | 1.4193          | 4.3921 | 18.8608 |
| 1.2936        | 4.0   | 1680 | 1.3936          | 4.5268 | 18.8363 |
| 1.1403        | 5.0   | 2100 | 1.4030          | 4.6813 | 18.8608 |
| 1.0284        | 6.0   | 2520 | 1.4078          | 4.8234 | 18.8684 |
| 1.0284        | 7.0   | 2940 | 1.4192          | 4.8126 | 18.8709 |
| 0.9196        | 8.0   | 3360 | 1.4303          | 4.8599 | 18.87   |
| 0.8459        | 9.0   | 3780 | 1.4506          | 4.7802 | 18.8599 |
| 0.7884        | 10.0  | 4200 | 1.4757          | 4.8895 | 18.8785 |
| 0.739         | 11.0  | 4620 | 1.4945          | 4.9126 | 18.8759 |
| 0.7083        | 12.0  | 5040 | 1.5058          | 4.9532 | 18.8709 |


### Framework versions

- Transformers 4.34.1
- Pytorch 2.1.0+cu118
- Datasets 2.14.6
- Tokenizers 0.14.1

## Resources
 - Christodouloupoulos, C., & Steedman, M. (2014). A massively parallel corpus: the Bible in 100 languages. Language Resources and Evaluation, 49(2), 375–395. https://doi.org/10.1007/s10579-014-9287-y
‌
 - Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., … Wei, J. (2022). _Scaling Instruction-Finetuned Language Models._ doi:10.48550/ARXIV.2210.11416
 
 - _Hawaii Pidgin_. (2017). Wycliffe. https://www.biblegateway.com/versions/Hawaii-Pidgin-HWP/ (Original work published 2000)

 - _King James Bible_. (2017). BibleGateway.com. https://www.biblegateway.com/versions/king-james-version-kjv-bible/ (Original work published 1769)

 - T5. (n.d.). Huggingface.co. https://huggingface.co/docs/transformers/model_doc/t5

 - Translation. (n.d.). Huggingface.co. Retrieved October 18, 2023, from https://huggingface.co/docs/transformers/tasks/translation
