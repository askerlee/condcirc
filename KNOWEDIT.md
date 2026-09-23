# KnowEdit in-context task

Download a JSON test file from [zjunlp/KnowEdit](https://huggingface.co/datasets/zjunlp/KnowEdit/tree/main/benchmark). Supported files are `benchmark/ZsRE/ZsRE-test-all.json`, `benchmark/wiki_counterfact/test_cf.json`, `benchmark/wiki_recent/recent_test.json`, and `benchmark/WikiBio/wikibio-test-all.json`.

For example, after downloading the ZsRE test file:

```sh
python infer.py --knowedit-file benchmark/ZsRE/ZsRE-test-all.json --knowedit-index 1-10 --max-new-tokens 64
```

Indices are 1-based; ranges and negative indices (counting from the end) work as for the other tasks. Omit `--knowedit-index` to run the entire file. Each output record includes `knowedit_source`, `knowedit_subject`, and `knowedit_target`; each run includes `knowedit_in_context_correct`, with a corresponding aggregate count in the summary. For fact and QA records, scoring requires the updated answer at the start of the response (case-insensitive, allowing trailing punctuation). WikiBio uses the reference sentence as a required beginning of the generated continuation.

This task provides the new fact in the prompt for fact/QA records and checks whether the model uses it. **It does not edit model weights or measure KnowEdit's edit success, portability, locality, or fluency metrics.** WikiBio uses passage continuation without supplying the target sentence. ConvSent and Sanitation have different schemas and are not supported by this task.