# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets
import deeplake

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 96

# takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
dataset = load_dataset("openwebtext")

# owt by default only contains the 'train' split, so create a test split
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

# this results in:
# >>> split_dataset
# DatasetDict({
#     train: Dataset({
#         features: ['text'],
#         num_rows: 8009762
#     })
#     val: Dataset({
#         features: ['text'],
#         num_rows: 4007
#     })
# })

enc = tiktoken.get_encoding("gpt2")

for split in split_dataset.keys():
    path = os.path.join(os.path.dirname(__file__), f'openwebtext-{split}')

    # define the dataset
    ds = deeplake.empty(path, overwrite=True)

    ds.create_tensor('text', htype="text", chunk_compression='lz4')
    ds.create_tensor('tokens', dtype=np.uint16, chunk_compression='lz4')

    @deeplake.compute
    def tokenize(example, ds):
        ids = enc.encode_ordinary(example) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        ds.append({"text": example, "tokens": np.array(ids).astype(np.uint16)})

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)

    tokenize().eval(split_dataset[split]['text'], ds, num_workers=num_proc, scheduler="processed")
    ds.commit()
    ds.summary()

# Dataset(path='./data/deeplake/openwebtext-train', tensors=['text', 'tokens'])
# tensor    htype           shape           dtype  compression
# -------  -------         -------         -------  ------- 
#  text     text        (8009762, 1)         str     None   
# tokens   generic  (8009762, 132:131288)  uint16    None   

#Dataset(path='./data/deeplake/openwebtext-val', tensors=['text', 'tokens'])
# tensor    htype         shape         dtype  compression
# -------  -------       -------       -------  ------- 
#  text     text        (4007, 1)        str     None   
# tokens   generic  (4007, 148:40801)  uint16    None  


# ./data/deeplake/openwebtext-val is ~35GB (tokens 15GB), ./deeplake/val ~21MB
# ./data/deeplake/openwebtext-val has ~9B tokens (9,035,582,198) and all text
# val has ~4M tokens (4,434,897)

# to read the bin files later, e.g. with numpy:
# ds = deeplake.load('./deeplake/openwebtext-train')