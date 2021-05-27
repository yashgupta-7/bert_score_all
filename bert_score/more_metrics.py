cnns_path = "/exp/yashgupta/ipsum/AdaptSum/dataset/dataset/dialogue/300sample/train.target" #"/exp/yashgupta/ipsum/AdaptSum/dataset/SDPT-cnn_dm/train.source"
cnnt_path = "/exp/yashgupta/ipsum/AdaptSum/dataset/SDPT-cnn_dm/train.target"
dias_path = "/exp/yashgupta/ipsum/AdaptSum/dataset/dataset/dialogue/300sample/train.target"
diat_path = "/exp/yashgupta/ipsum/AdaptSum/dataset/dataset/dialogue/300sample/train.source"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import importlib
import bert_score
importlib.reload(bert_score)

with open(cnns_path) as f: #"example/hyps.txt"
    cands = [line.strip() for line in f]

with open(dias_path) as f: #"example/refs.txt"
    refs = [line.strip() for line in f]

P, R, F1 = bert_score.score_all(cands, refs, lang='en', verbose=True, batch_size=64)
import torch
torch.save(F1, 'cnn_dia_F1.pt')