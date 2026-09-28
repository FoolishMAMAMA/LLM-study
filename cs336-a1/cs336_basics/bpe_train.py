import regex


def train_bpe(input_path:str,vocab_size:int,special_tokens:list[str])->tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    vocab = dict()
    for i in range(256):
        vocab[i]=bytes([i])
    
    for i in range(256,256+len(special_tokens)):
        vocab[i]= special_tokens[i-256].encode("utf-8")

    merges = []

    with open(input_path,"r",encoding="utf-8") as f:
        text = f.read()
        print(repr(text[:200]))

    return vocab, merges



vocab, merges = train_bpe(
    "tests/fixtures/corpus.en", 300, ["<|endoftext|>"]
)

print(len(vocab))
print(vocab[256])
print(merges)