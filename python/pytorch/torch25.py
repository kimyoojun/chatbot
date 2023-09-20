import torch
import torchtext
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
from torchtext import datasets

start = time.time()
TEXT = torchtext.data.Field(lower=True, fix_length=200, batch_first=False)
LABEL = torchtext.data.Field(sequential=False)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

print(vars(train_data.examples[0]))