from minbpe.base import Tokenizer


def get_stats(ids,count=None):
    count = {} if count is None else count
    for pair in zip(ids,ids[1:]):
        count[pair] = count.get(pair,0)+1
    return count
                            


def merge(ids, pair, idx):
    newids=[ ]
    i = 0
    while i< len(ids):
        if(i<len(ids)-1 and ids[i]==pair[0] and ids[i+1]==pair[1]):
            newids.append(idx)
            i =i+2
        else:
            newids.append(ids[i])
            i = i+1
    return newids


class MyBasicTokenizer(Tokenizer):

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size>=256
        number_merges = vocab_size-256

        text_bytes = text.encode("utf-8")
        text_ids = list(text_bytes)
        print("Text_ids is: ",text_ids)
        merges ={}
        vocab = {idx:bytes([idx]) for idx in range(256)}
        for i in range(number_merges):
            stats = get_stats(text_ids)
            target_pair = max(stats, key=stats.get)
            idx = 256+i
            text_ids=   merge(text_ids,target_pair,idx)
            merges[target_pair]=idx 
            vocab[idx] = vocab[target_pair[0]]+vocab[target_pair[1]]
            if (verbose):
                print(f"merge {i+1}/{number_merges}: {target_pair} -> {idx} ({vocab[idx]}) had {stats[target_pair]} occurrences")
        self.vocab=vocab#use in decode
        self.merges=merges#use in encode

tokenizer = MyBasicTokenizer()
tokenizer.train("abab", vocab_size=258, verbose=True)

print(tokenizer.merges)
print(tokenizer.vocab[256])
print(tokenizer.vocab[257])

assert tokenizer.merges == {
    (97, 98): 256,
    (256, 256): 257,
}

assert tokenizer.vocab[256] == b"ab"
assert tokenizer.vocab[257] == b"abab"