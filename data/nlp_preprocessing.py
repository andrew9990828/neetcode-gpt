import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        pos = []
        neg = []

        for n in negative:
            neg.append(n.split())
        
        for p in positive:
            pos.append(p.split())
        
        comb = pos + neg

        all_words = []

        for r in range(len(comb)):
            for w in comb[r]:
                all_words.append(w)
        
        words = set(all_words)
        words = sorted(words)

        vocab = {}

        for i, n in enumerate(words):
            vocab[n] = vocab.get(n, i+1)
        
        for i in range(len(pos)):
            for j in range(len(pos[i])):
                pos[i][j] = vocab[pos[i][j]]
                
        for i in range(len(neg)):
            for j in range(len(neg[i])):
                neg[i][j] = vocab[neg[i][j]]

        seqs = []

        for sentence in pos + neg:
            seqs.append(torch.tensor(sentence))

        output = nn.utils.rnn.pad_sequence(seqs, batch_first=True)

        return output.float()
