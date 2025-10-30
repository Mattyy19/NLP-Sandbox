---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:29352
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L12-v2
widget:
- source_sentence: The New York Times
  sentences:
  - They take advantage of factors that reduce visibility; many kills take place near
    some form of cover or at night. One study in 2018 recorded a lion running at a
    top speed of 74.1 km/h (46.0 mph). The lion accelerates at the start of the chase
    by a rate of 9.5 m/s (34 km/h; 21 mph) per second, whereas zebras, wildebeest
    and Thomson's gazelle accelerate by a rate of 5 m/s (18 km/h; 11 mph) per second,
    5.6 m/s (20 km/h; 13 mph) per second, and 4.5 m/s (16 km/h; 10 mph) per second
    respectively; acceleration appears to be more important than steady displacement
    speed in lion hunts.
  - He moved across the river to Southwark by 1599, the same year his company constructed
    the Globe Theatre there. By 1604 he had moved north of the river again, to an
    area north of St Paul's Cathedral with many fine houses. There he rented rooms
    from a French Huguenot named Christopher Mountjoy, a maker of women's wigs and
    other headgear.
  - Retrieved January 11, 2024. "New Roles for Emily Cochrane and Campbell Robertson".
    The New York Times Company.
- source_sentence: Leopard
  sentences:
  - It is not known whether treating other sexually transmitted infections is effective
    in preventing HIV. Pre-exposure Antiretroviral treatment among people with HIV
    whose CD4 count ≤ 550 cells/μL is a very effective way to prevent HIV infection
    of their partner (a strategy known as treatment as prevention, or TASP). TASP
    is associated with a 10- to 20-fold reduction in transmission risk.
  - As in 1977, there were street parties and commemorative events, and monuments
    were named to honour the occasion. One million people attended each day of the
    three-day main Jubilee celebration in London, and the enthusiasm shown for Elizabeth
    by the public was greater than many journalists had anticipated. In 2003, Elizabeth
    sued the Daily Mirror for breach of confidence and obtained an injunction which
    prevented the outlet from publishing information gathered by a reporter who posed
    as a footman at Buckingham Palace.
  - 'Gland, Switzerland: IUCN/SSC Cat Specialist Group. Archived from the original
    on 2014-02-22. ^ a b Schütze, H. (2002).'
- source_sentence: Protein
  sentences:
  - doi:10.4141/S01-054. ^ Radzicka A, Wolfenden R (1 January 1996). "Rates of Uncatalyzed
    Peptide Bond Hydrolysis in Neutral Solution and the Transition State Affinities
    of Proteases".
  - 'Two months later, Gaga attended the 84th Annual US Conference of Mayors in Indianapolis
    where together with the Dalai Lama she talked about the power of kindness and
    how to make the world a more compassionate place. In April 2020, Gaga curated
    the televised benefit concert, One World: Together at Home, a collaboration with
    Global Citizen to benefit the World Health Organization''s COVID-19 Solidarity
    Response Fund. The special raised $127 million, which according to Forbes "puts
    it on par with the other legendary fundraiser, Live Aid, as the highest grossing
    charity concert in history."'
  - If not disturbed, they may drink continuously for ten minutes. Due to the scarcity
    of water sources, emus are sometimes forced to go without water for several days.
    In the wild, they often share water holes with other animals such as kangaroos;
    they are wary and tend to wait for the other animals to leave before drinking.
- source_sentence: Birmingham
  sentences:
  - This can create coverage issues in the administration of no-fault insurance schemes
    such as workers' compensation. In general, a heart attack is not covered; however,
    it may be a work-related injury if it results, for example, from unusual emotional
    stress or unusual exertion. In addition, in some jurisdictions, heart attacks
    had by persons in particular occupations such as police officers may be classified
    as line-of-duty injuries by statute or policy.
  - '"The studios where Joe Lycett''s new Channel 4 show is filmed". Birmingham Live.
    Retrieved 2 July 2024.'
  - Retrieved 3 April 2025. ^ "Syrian Arab Republic (SYR) – Demographics, Health &
    Infant Mortality". UNICEF DATA.
- source_sentence: George IV
  sentences:
  - S2CID 35030583 – via SpringerLink. ^ "Hazardous Substances and New Organisms Act
    2003 – Schedule 2 Prohibited new organisms". New Zealand Government.
  - During the 1980s, films such as A Better Tomorrow, As Tears Go By, and Zu Warriors
    from the Magic Mountain expanded global interest beyond martial arts films; locally
    made gangster films, romantic dramas, and supernatural fantasies became popular.
    Hong Kong cinema continued to be internationally successful over the following
    decade with critically acclaimed dramas such as Farewell My Concubine, To Live,
    and Chungking Express. The city's martial arts film roots are evident in the roles
    of the most prolific Hong Kong actors.
  - For political reasons, the union was to remain secret and Fitzherbert promised
    not to reveal it. But, in spring 1786, covert allusions to the marriage appeared
    in the press, and several satirical prints depicted the clandestine marriage.
    Prince George was plunged into debt by his exorbitant lifestyle.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L12-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) <!-- at revision c004d8e3e901237d8fa7e9fff12774962e391ce5 -->
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
  (2): Normalize()
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
    'George IV',
    'For political reasons, the union was to remain secret and Fitzherbert promised not to reveal it. But, in spring 1786, covert allusions to the marriage appeared in the press, and several satirical prints depicted the clandestine marriage. Prince George was plunged into debt by his exorbitant lifestyle.',
    "During the 1980s, films such as A Better Tomorrow, As Tears Go By, and Zu Warriors from the Magic Mountain expanded global interest beyond martial arts films; locally made gangster films, romantic dramas, and supernatural fantasies became popular. Hong Kong cinema continued to be internationally successful over the following decade with critically acclaimed dramas such as Farewell My Concubine, To Live, and Chungking Express. The city's martial arts film roots are evident in the roles of the most prolific Hong Kong actors.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.7277, -0.0846],
#         [ 0.7277,  1.0000,  0.0161],
#         [-0.0846,  0.0161,  1.0000]])
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
  | details | <ul><li>min: 3 tokens</li><li>mean: 4.25 tokens</li><li>max: 11 tokens</li></ul> | <ul><li>min: 22 tokens</li><li>mean: 64.04 tokens</li><li>max: 128 tokens</li></ul> |
* Samples:
  | sentence_0                          | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
  |:------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>List of Roman emperors</code> | <code>Died of an intestinal disease AlexanderΑλέξανδρος 11 May 912 – 6 June 913(1 year and 26 days) Son of Basil I; co-emperor since September or October 879 23 November 870 – 6 June 913(aged 42)Died of illness, possibly testicular cancer Constantine VIIPorphyrogenitusΚωνσταντῖνος 6 June 913 – 9 November 959(46 years, 5 months and 3 days) Son of Leo VI; co-emperor since 15 May 908. Successively dominated by regents and co-emperors until 27 January 945, when he deposed Romanos I's sons 17/18 May 905 – 9 November 959(aged 54)Saw the beginning of renewed expansion in the East against the Arabs. Remembered for his numerous writings.</code> |
  | <code>Evolution</code>              | <code>Their discoveries have influenced not just the development of biology but also other fields including agriculture, medicine, and computer science. Heredity Evolution in organisms occurs through changes in heritable characteristics—the inherited characteristics of an organism. In humans, for example, eye colour is an inherited characteristic and an individual might inherit the "brown-eye trait" from one of their parents.</code>                                                                                                                                                                                                                |
  | <code>Michael Jackson</code>        | <code>This transaction is possibly the largest for a single musician's work. Posthumous releases and productions Jackson's posthumous releases and productions are administered by the estate of Michael Jackson, which owns Jackson's trademarks and rights to his name, image and likeness. The first posthumous Jackson song, "This Is It", co-written in the 1980s with Paul Anka, was released in October 2009.</code>                                                                                                                                                                                                                                         |
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
| 0.1363 | 500  | 0.5727        |
| 0.2726 | 1000 | 0.5398        |
| 0.4088 | 1500 | 0.461         |
| 0.5451 | 2000 | 0.4155        |
| 0.6814 | 2500 | 0.3839        |
| 0.8177 | 3000 | 0.3379        |
| 0.9539 | 3500 | 0.3528        |


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