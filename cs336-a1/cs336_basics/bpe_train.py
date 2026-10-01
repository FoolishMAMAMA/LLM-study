import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def train_bpe(input_path:str,vocab_size:int,special_tokens:list[str])->tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    vocab = dict()
    for i in range(256):
        vocab[i]=bytes([i])
    
    for i in range(256,256+len(special_tokens)):
        vocab[i]= special_tokens[i-256].encode("utf-8")

    merges = []

    with open(input_path,"r",encoding="utf-8") as f:
        text = f.read()
        #print(repr(text[:200]))
        if special_tokens:
            escaped_tokens=[re.escape(t) for t in special_tokens]
            pattern = "|".join(escaped_tokens)
            segments = re.split(pattern,text)
        else:
            segments = [text]
        # print(segments[:3])
        #print("片段数：", len(segments))
        #print("第一个片段的前100个字符：", repr(segments[0][:100]))
    
    # for match in re.finditer(PAT, "hello world!"):
    #     print(repr(match.group()))

    frag=dict()
    for i in segments:
        for match in re.finditer(PAT,i):
            token = match.group()
            frag[token] = frag.get(token, 0) + 1
    print(frag)

    word_counts = {}
    for key, value  in frag.items():
        #seperate=tuple(ch.encode("utf-8") for ch in key)错误写法，只支持英文！
        separate = tuple(bytes([b]) for b in key.encode("utf-8"))
        word_counts[separate] = value

    pair_counts = {}
    for item,frequency in word_counts.items():
        if len(item)<2:
            continue
        else:
            for i in range(len(item)-1):
                k_item  = (item[i],item[i+1])
                pair_counts[k_item]= pair_counts.get(k_item,0)+frequency
    print(word_counts)
    print(pair_counts)
    return vocab, merges


# print("测试零：打开文件f.read()")
# vocab, merges = train_bpe(
#     "tests/fixtures/corpus.en", 300, ["<|endoftext|>"]
# )

# print(len(vocab))
# print(vocab[256])
# print(merges)

# print("测试一：没有特殊 token")
# train_bpe("toy.txt", 300, [])


# print("测试二：有特殊 token")
# train_bpe("toy.txt", 300, ["<|endoftext|>"])

# print("测试三：预分词")
# train_bpe("toy_hello.txt", 300, ["<|endoftext|>"])
print("测试四：临对频率")
train_bpe("toy_pair_count.txt", 300, ["<|endoftext|>"])