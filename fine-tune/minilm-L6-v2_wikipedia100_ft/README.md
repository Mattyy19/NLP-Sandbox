---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:29352
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Tiger
  sentences:
  - '{{cite journal}}: CS1 maint: article number as page number (link) ^ Akopyants
    NS, Kimblin N, Secundino N, Patrick R, Peters N, Lawyer P, Dobson DE, Beverley
    SM, Sacks DL (April 2009). "Demonstration of genetic exchange during cyclical
    development of Leishmania in the sand fly vector". Science.'
  - Nichols would claim they sought him to violate New Jersey's blue law (a restriction
    common in South Jersey and Pennsylvania as a remnant of the influence of their
    Quakers roots). McCall requested ginger ales as non-alcoholic beverages were not
    subject to the blue law. Nichols refused the group even ginger ales and reportedly
    stated "the best thing would be for you to leave".
  - ^ Mazák, V.; Groves, C. P. & Van Bree, P. (1978). "Skin and Skull of the Bali
    Tiger, and a list of preserved specimens of Panthera tigris balica (Schwarz, 1912)".
    Zeitschrift für Säugetierkunde.
- source_sentence: Elizabeth II
  sentences:
  - Archived from the original (PDF) on October 7, 2016. Retrieved August 11, 2016.
    ^ Okuyama, Minami W.; et al.
  - '^ Sibley, Charles; Jon Edward Ahlquist (1990). Phylogeny and classification of
    birds. New Haven: Yale University Press.'
  - Elizabeth II Elizabeth II Elizabeth IIHead of the CommonwealthFormal portrait,
    1959Queen of the United Kingdomand other Commonwealth realms Reign6 February 1952
    – 8 September 2022Coronation2 June 1953PredecessorGeorge VISuccessorCharles IIIBornPrincess
    Elizabeth of York(1926-04-21)21 April 1926Mayfair, London, EnglandDied8 September
    2022(2022-09-08) (aged 96)Balmoral Castle, Aberdeenshire, ScotlandBurial19 September
    2022King George VI Memorial Chapel, St George's Chapel, Windsor CastleSpouse Prince
    Philip, Duke of Edinburgh ​ ​(m. 1947; died 2021)​IssueDetail Charles III Anne,
    Princess Royal Prince Andrew, Duke of York Prince Edward, Duke of Edinburgh NamesElizabeth
    Alexandra MaryHouseWindsorFatherGeorge VIMotherElizabeth Bowes-LyonReligionProtestant
    Signature Elizabeth II (Elizabeth Alexandra Mary; 21 April 1926 – 8 September
    2022) was Queen of the United Kingdom and other Commonwealth realms from 6 February
    1952 until her death in September 2022. She had been queen regnant of 32 sovereign
    states during her lifetime and was the monarch of 15 realms at her death. Her
    reign of 70 years and 214 days is the longest of any British monarch, the second-longest
    of any sovereign state, and the longest of any queen regnant in history.
- source_sentence: Anfield
  sentences:
  - '^ a b "Anfield: the victims, the anger and Liverpool''s shameful truth | David
    Conn". the Guardian. 6 May 2013.'
  - '"Football: You only sing when you''re standing". London: Independent, The. Archived
    from the original on 11 November 2012.'
  - 'August 10, 2010. Retrieved May 31, 2015. ^ "Exclusive: Will.i.am Explains His
    ''Disgust'' for New Michael Jackson Album".'
- source_sentence: YouTube
  sentences:
  - Archived from the original on 17 January 2022. Retrieved 17 January 2022. ^ Department
    of Statistics Malaysia (2021).
  - Retrieved April 6, 2017. ^ Biggs, John (May 4, 2007). "YouTube Launches Revenue
    Sharing Partners Program, but no Pre-Rolls".
  - B2022. ^ a b Gardner, Jamie (6 August 2022). "Birmingham could host Olympics having
    'totally embraced' Commonwealth Games".
- source_sentence: Wales
  sentences:
  - '^ Hernández, Javier C. (30 June 2020). "Harsh Penalties, Vaguely Defined Crimes:
    Hong Kong''s Security Law Explained". The New York Times.'
  - ^ Taylor, John D.; Glover, Emily A.; Williams, Suzanne T. (2009). "Phylogenetic
    position of the bivalve family Cyrenoididae – removal from (and further dismantling
    of) the superfamily Lucinoidea". Nautilus.
  - ^ Williams, Nino (25 November 2018). "The uncomfortable truth about the three
    feathers symbol embraced by Wales". WalesOnline.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
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
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
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
    'Wales',
    '^ Williams, Nino (25 November 2018). "The uncomfortable truth about the three feathers symbol embraced by Wales". WalesOnline.',
    '^ Taylor, John D.; Glover, Emily A.; Williams, Suzanne T. (2009). "Phylogenetic position of the bivalve family Cyrenoididae – removal from (and further dismantling of) the superfamily Lucinoidea". Nautilus.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6771, 0.0571],
#         [0.6771, 1.0000, 0.1031],
#         [0.0571, 0.1031, 1.0000]])
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
  |         | sentence_0                                                                       | sentence_1                                                                         |
  |:--------|:---------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                             |
  | details | <ul><li>min: 3 tokens</li><li>mean: 4.31 tokens</li><li>max: 11 tokens</li></ul> | <ul><li>min: 21 tokens</li><li>mean: 68.8 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | sentence_0                             | sentence_1                                                                                                                                                           |
  |:---------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>List of Roman emperors</code>    | <code>^ ODB, pp. 1366–1367; Schreiner, p. 157–159; Norwich 1993, p. 361. ^ ODB, p. 1479; Schreiner, p. 158–159; Grierson 1973, p. 798–799, 821; Maynard 2021.</code> |
  | <code>Human</code>                     | <code>PMID 21427751. ^ Weatherall DJ (May 2008). "Genetic variation and susceptibility to infection: the red cell and malaria".</code>                               |
  | <code>Voting Rights Act of 1965</code> | <code>^ Bravin, Jess (June 23, 2009). "Supreme Court Avoids Voting-Rights Act Fight". The Wall Street Journal.</code>                                                |
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
| 0.1363 | 500  | 0.6353        |
| 0.2726 | 1000 | 0.557         |
| 0.4088 | 1500 | 0.5117        |
| 0.5451 | 2000 | 0.4392        |
| 0.6814 | 2500 | 0.4432        |
| 0.8177 | 3000 | 0.4035        |
| 0.9539 | 3500 | 0.3785        |


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