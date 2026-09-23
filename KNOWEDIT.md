# KnowEdit target-token evaluation

Download a JSON test file from [zjunlp/KnowEdit](https://huggingface.co/datasets/zjunlp/KnowEdit/tree/main/benchmark). Supported files are `benchmark/ZsRE/ZsRE-test-all.json`, `benchmark/wiki_counterfact/test_cf.json`, `benchmark/wiki_recent/recent_test.json`, and `benchmark/WikiBio/wikibio-test-all.json`.

For example, after downloading the ZsRE test file:

```sh
python infer.py --knowedit-file benchmark/ZsRE/ZsRE-test-all.json --knowedit-index 1-10 --max-new-tokens 64
```

Indices are 1-based; ranges and negative indices (counting from the end) work as for the other tasks. Omit `--knowedit-index` to run the entire file. Each output record includes `knowedit_target` (`target_new`) and, where provided, `knowedit_ground_truth`. Each run scores both independently using teacher-forced next-token accuracy: `knowedit_ground_truth_acc` for the original answer and `knowedit_target_new_acc` for the proposed edit. The terminal prints the top two predictions and their full-vocabulary softmax probabilities at each token position for each available reference, regardless of whether they match. Summary means show how many selected records had that reference. ZsRE and WikiCounterFact provide `ground_truth`; WikiRecent and WikiBio do not, so they receive only the target-new score (WikiBio uses `labels` as the continuation target). A fresh model cache uses the same recirculation settings as the corresponding run; each reference token is fed only after its prediction has been scored. `--do-eval` does not call OpenAI or a separate evaluation model for KnowEdit.

Fact and QA generation prompts contain only the question or completion statement, never either reference answer. During teacher-forced scoring, gold tokens are present in the model input only *after* their predictions have been measured. Neither score assumes the model is edited: compare the two accuracies for the model you actually ran. **Without an editing step and pre/post comparison, these scores are not KnowEdit edit success, portability, or locality.** Dynamic recovery options triggered by freely generated text are not replayed during teacher-forced scoring. ConvSent and Sanitation have different schemas and are not supported by this task.