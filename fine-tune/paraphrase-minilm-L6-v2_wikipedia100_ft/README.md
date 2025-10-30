---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:29352
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/paraphrase-MiniLM-L6-v2
widget:
- source_sentence: Old Trafford
  sentences:
  - U.S. National Library of Medicine. Archived from the original on August 5, 2011.
    Retrieved August 17, 2011.
  - References Citations ^ "BBC Wales – Music – National Anthem – The background to
    Hen Wlad Fy Nhadau". www.bbc.co.uk. Retrieved 27 April 2024.
  - Retrieved 25 March 2020. ^ O'Brien, James (17 October 2020). "Hull considered
    as potential Grand Final host as Super League searches for Old Trafford alternative".
- source_sentence: World Heritage Site
  sentences:
  - The nails of burrowing species tend to be long and strong, while arboreal rodents
    have shorter, sharper nails. Rodenta, have nails on their first digit which they
    use in manual food handling. Such a nail combined with dexterous feeding movement
    with incisors which allow them to eat hard seeds and nuts, a niche that they presently
    dominate.
  - 'The Dynamics of Ancient Empires: State Power from Assyria to Byzantium. Oxford:
    Oxford University Press. ISBN 978-0-19-537158-1 ^ Tuell 1991, p. 51.'
  - ^ Slezak, Michael (26 May 2016). "Australia scrubbed from UN climate change report
    after government intervention". The Guardian.
- source_sentence: Rodent
  sentences:
  - ISBN 978-0-8247-5407-5. ^ Harkness, John E.; Wagner, Joseph E. (1995). The Biology
    and Medicine of Rabbits and Rodents.
  - Heat sensors in the nose help them to detect blood vessels near the surface of
    the skin. They pierce the animal's skin with their teeth, biting away a small
    flap, and lap up the blood with their tongues, which have lateral grooves adapted
    to this purpose. The blood is kept from clotting by an anticoagulant in the saliva.
  - In 2021, international scientists recommended UNESCO to put the Great Barrier
    Reef on the endangered list, as global climate change had caused a further negative
    state of the corals and water quality. Again, the Australian government campaigned
    against this, and in July 2021, the World Heritage Committee, made up of diplomatic
    representatives of 21 countries, ignored UNESCO's assessment, based on studies
    of scientists, "that the reef was clearly in danger from climate change and so
    should be placed on the list." According to environmental protection groups, this
    "decision was a victory for cynical lobbying and Australia, as custodians of the
    world's biggest coral reef, was now on probation."
- source_sentence: Syria
  sentences:
  - Royal assent was finally granted to the Catholic Relief Act on 13 April. Declining
    health and death King Henry IV by William Heath, c. 1827Henry IV Part 2 Act II
    Scene 4 by Henry Fuseli, 1805Cartoon (left) of George IV and his mistress Lady
    Conyngham, satirised as John Falstaff and Doll Tearsheet, mirroring a well known
    work (right) by Fuseli George's heavy drinking and indulgent lifestyle had taken
    their toll on his health by the late 1820s. While still Prince of Wales, he had
    become obese through his huge banquets and copious consumption of alcohol, making
    him the target of ridicule on the rare occasions that he appeared in public; by
    1797, his weight had reached 17 stone 7 pounds (111 kg; 245 lb).
  - The coastal mountainous region was occupied in part by the Nizari Ismailis, the
    so-called Assassins, who had intermittent confrontations and truces with the Crusader
    States. Later in history when "the Nizaris faced renewed Frankish hostilities,
    they received timely assistance from the Ayyubids." After a century of Seljuk
    rule, Syria was largely conquered (1175–1185) by the Kurdish liberator Salah ad-Din,
    founder of the Ayyubid dynasty of Egypt.
  - ^ Livingstone, Robert (11 November 2011). "London Defeats Doha to host 2017 International
    Athletics Championships". Gamesbids.com.
- source_sentence: Red fox
  sentences:
  - These variable performances have divided critics. Richard Williams and Andy Gill
    argued that Dylan has found a successful way to present his rich legacy of material.
    Others have criticized his live performances for changing "the greatest lyrics
    ever written so that they are effectively unrecognisable", and giving so little
    to the audience that "it is difficult to understand what he is doing on stage
    at all".
  - PMID 12778049. ^ a b Crittenden AN, Schnorr SL (2017). "Current views on hunter-gatherer
    nutrition and the evolution of the human diet".
  - The gestation period lasts 49–58 days. Though foxes are largely monogamous, DNA
    evidence from one population indicated large levels of polygyny, incest and mixed
    paternity litters. Subordinate vixens may become pregnant, but usually fail to
    whelp, or have their kits killed postpartum by either the dominant female or other
    subordinates.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/paraphrase-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2) <!-- at revision c9a2bfebc254878aee8c3aca9e6844d5bbb102d1 -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Red fox',
    'The gestation period lasts 49–58 days. Though foxes are largely monogamous, DNA evidence from one population indicated large levels of polygyny, incest and mixed paternity litters. Subordinate vixens may become pregnant, but usually fail to whelp, or have their kits killed postpartum by either the dominant female or other subordinates.',
    'PMID 12778049. ^ a b Crittenden AN, Schnorr SL (2017). "Current views on hunter-gatherer nutrition and the evolution of the human diet".',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6035, 0.3208],
#         [0.6035, 1.0000, 0.4789],
#         [0.3208, 0.4789, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 29,352 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                       | sentence_1                                                                          |
  |:--------|:---------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                              |
  | details | <ul><li>min: 3 tokens</li><li>mean: 4.37 tokens</li><li>max: 11 tokens</li></ul> | <ul><li>min: 19 tokens</li><li>mean: 64.06 tokens</li><li>max: 128 tokens</li></ul> |
* Samples:
  | sentence_0                    | sentence_1                                                                                                                                                                                                                                                                                                                                                         |
  |:------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Amazon (company)</code> | <code>Retrieved July 7, 2019. ^ "Amazon announces 2 new ways it's using robots to assist employees and deliver for customers". US About Amazon.</code>                                                                                                                                                                                                             |
  | <code>Animal</code>           | <code>In most cases, a third germ layer, the mesoderm, also develops between them. These germ layers then differentiate to form tissues and organs. Repeated instances of mating with a close relative during sexual reproduction generally leads to inbreeding depression within a population due to the increased prevalence of harmful recessive traits.</code> |
  | <code>Lady Gaga</code>        | <code>Archived from the original on July 12, 2015. Retrieved June 29, 2015.Howard, Caroline (August 24, 2011). "The World's 100 Most Powerful Women: This Year It's All About Reach".</code>                                                                                                                                                                       |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.1363 | 500  | 0.7787        |
| 0.2726 | 1000 | 0.6933        |
| 0.4088 | 1500 | 0.5905        |
| 0.5451 | 2000 | 0.5639        |
| 0.6814 | 2500 | 0.5015        |
| 0.8177 | 3000 | 0.4335        |
| 0.9539 | 3500 | 0.4277        |


### Framework Versions
- Python: 3.10.0
- Sentence Transformers: 5.1.2
- Transformers: 4.57.1
- PyTorch: 2.9.0+cpu
- Accelerate: 1.11.0
- Datasets: 4.2.0
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->