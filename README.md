# Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach

## Description
Zero-shot text classification (0SHOT-TC) is a challenging NLU problem to which little attention has been paid by the research community. 0SHOT-TC aims to associate an appropriate label with a piece of text, irrespective of the text domain and the aspect (e.g., topic, emotion, event, etc.) described by the label. And there are only a few articles studying 0SHOT-TC, all focusing only on topical categorization which, we argue, is just the tip of the iceberg in 0SHOT-TC. In addition, the chaotic experiments in literature make no uniform comparison, which blurs the progress. This work benchmarks the 0SHOT-TC problem by providing unified datasets, standardized evaluations, and state-of-the-art baselines. Our contributions include: i) The datasets we provide facilitate studying 0SHOT-TC relative to conceptually different and diverse aspects: the “topic” aspect includes “sports” and “politics” as labels; the “emotion” aspect includes “joy” and “anger”; the “situation” aspect includes “medical assistance” and “water shortage”. ii) We extend the existing evaluation setup (labelpartially-unseen) – given a dataset, train on some labels, test on all labels – to include a more challenging yet realistic evaluation label-fully-unseen 0SHOT-TC, aiming at classifying text snippets without seeing task specific training data at all. iii) We unify the 0SHOT-TC of diverse aspects within a textual entailment formulation and study it this way.
Hi, this repository contains the code and the data for the EMNLP2019 paper "Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach"

## Datasets
Dataset for "topic detection", "emotion detection" and "situation detection" 
- https://drive.google.com/open?id=1qGmyEVD19ruvLLz9J0QGV7rsZPFEz2Az

Wikipedia data and three pretrained entailment models (RTE, MNLI, FEVER)
- https://drive.google.com/file/d/1ILCQR_y-OSTdgkz45LP7JsHcelEsvoIn/view?usp=sharing


## Requirement
- Pytorch
- Transformer (pytorch): https://github.com/huggingface/transformers
- GPU

To rerun the code (take "baseline_wiki_based_emotion.py" as an example):

    CUDA_VISIBLE_DEVICES=1 python -u baseline_wiki_based_emotion.py --task_name rte --do_train --do_lower_case --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --data_dir '' --output_dir ''

Very important step before running:
Since our code was written in "pytorch-transformer" -- the old verion of Huggingface Transformer, pls
1) update the "pytorch-transformer" into "transformer" before running the code. For example:

Now it is:

    from pytorch_transformers.file_utils import PYTORCH_TRANSFORMERS_CACHE
    from pytorch_transformers.modeling_bert import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
    from pytorch_transformers.tokenization_bert import BertTokenizer
    from pytorch_transformers.optimization import AdamW

change  to be:

    from transformers.file_utils import PYTORCH_TRANSFORMERS_CACHE
    from transformers.modeling_bert import BertForSequenceClassification
    from transformers.tokenization_bert import BertTokenizer
    from transformers.optimization import AdamW

2) the new Transformer's function "BertForSequenceClassification" has parameter order slightly different with the prior "pytorch_transformer". The current version is:

        def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                    position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

 the old version is:

       def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,
                  position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

 namely the token_ids and mask are exchanged. So, you need to change the input order (only for "token_type_ids" and "attention_mask") when ever you call the model. For example, my code currently is:

             logits = model(input_ids, input_mask,segment_ids, labels=None)

 change it to be:

             logits = model(input_ids, segment_ids, input_mask, labels=None)
             
## Citation 
For code and data: 

    @inproceedings{yinroth2019zeroshot,
        title={Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach},
        author={Wenpeng Yin, Jamaal Hay and Dan Roth},
        booktitle={{EMNLP}},
        url = {https://arxiv.org/abs/1909.00161},
        year={2019}
    }
## Contacts

For any questions : mr.yinwenpeng@gmail.com
