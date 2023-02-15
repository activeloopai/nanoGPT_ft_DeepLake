
## openwebtext Deep Lake dataset

after running `prepare.py` (preprocess) we get:

- ./deeplake/train is (~32GB, where tokens tensor is 15GB), ./deeplake/val ~21MB
- ./deeplake/train has ~9B tokens (9,035,582,198)
- val has ~4M tokens (4,434,897)

this came from 8,013,769 documents in total.

to move to your Activeloop account

```
import deeplake
token = '...'
deeplake.deepcopy('./data/deeplake/openwebtext-train/', 'hub://activeloop/openwebtext-train', token=token, scheduler='processed', num_workers=16)
```

references:

- OpenAI's WebText dataset is discussed in [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) dataset
