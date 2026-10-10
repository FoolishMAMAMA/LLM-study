from cs336_basics.bpe_pretokenization import DEFAULT_CHUNK_BYTES, PAT as PAT, build_word_counts

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_processes: int = 1,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    algorithm: str = "indexed",
    progress=None,
    checkpoint_path=None,
    checkpoint_interval: float = 300,
    resume_path=None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if algorithm == "indexed":
        from cs336_basics.bpe_indexed import train_bpe_indexed

        return train_bpe_indexed(
            input_path, vocab_size, special_tokens,
            num_processes=num_processes, chunk_bytes=chunk_bytes,
            progress=progress, checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval, resume_path=resume_path,
        )
    if algorithm != "reference":
        raise ValueError("algorithm 必须为 indexed 或 reference")
    if checkpoint_path is not None or resume_path is not None:
        raise ValueError("检查点仅支持 indexed 算法")
    vocab = dict()
    for i in range(256):
        vocab[i]=bytes([i])
    preserved_len = 256+len(special_tokens)
    if(vocab_size<256+len(special_tokens)):
        raise ValueError(f"vocab_size 至少为 {preserved_len}，实际收到 {vocab_size}")
    for i in range(256,256+len(special_tokens)):
            vocab[i]= special_tokens[i-256].encode("utf-8")

    merges = []

    word_counts = build_word_counts(
        input_path, special_tokens, num_processes=num_processes, chunk_bytes=chunk_bytes
    )

    pair_counts = count_pairs(word_counts)
    for i in range(vocab_size-preserved_len):

        if len(pair_counts)==0:
            break
        fittest_pair = find_fittest_pair(pair_counts)
        merges.append(fittest_pair)
        vocab[i+preserved_len] = fittest_pair[0]+fittest_pair[1]
        word_counts = merge_word_counts(word_counts,fittest_pair,pair_counts)
    
    return vocab, merges


def pair_diff(
    old_pair: dict[tuple, int],
    new_pair: dict[tuple, int],
    pair_counts: dict[tuple, int],
) -> None:
    """用已加权的旧、新贡献更新全局计数，包括已经消失的 pair。"""
    for pair in old_pair.keys() | new_pair.keys():
        delta = new_pair.get(pair, 0) - old_pair.get(pair, 0)
        if delta == 0:
            continue
        updated_count = pair_counts.get(pair, 0) + delta
        if updated_count < 0:
            raise ValueError(f"pair {pair!r} 的计数变成负数，请检查旧贡献")
        if updated_count == 0:
            pair_counts.pop(pair, None)
        else:
            pair_counts[pair] = updated_count





def merge_pair(
    tokens: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    if pair[0] not in tokens or pair[1] not in tokens:
        return tokens
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            new_tokens.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return tuple(new_tokens)


def count_pairs(word_counts:dict[tuple,int])->dict[tuple,int]:
    pair_counts = {}
    for item,frequency in word_counts.items():
            if len(item)<2:
                continue
            else:
                for i in range(len(item)-1):
                    k_item  = (item[i],item[i+1])
                    pair_counts[k_item]= pair_counts.get(k_item,0)+frequency

    #print(pair_counts)
    return pair_counts

def find_fittest_pair(pair_counts:dict[tuple,int])->tuple[bytes,bytes]:
    key,value = max(pair_counts.items(),key=lambda kv:(kv[1],kv[0]))
    #print(key,value)
    return key

def merge_word_counts(
    word_counts: dict[tuple, int],
    fittest_pair: tuple[bytes, bytes],
    pair_counts: dict[tuple, int],
) -> dict[tuple, int]:
    """返回合并后的频次表，并就地更新全局 pair 计数。"""
    new_word_counts={}
    for token,count in word_counts.items():
        merged_tokens = merge_pair(token,fittest_pair)
        new_word_counts[merged_tokens]=new_word_counts.get(merged_tokens,0)+count
        if merged_tokens != token:
            # 只统计变化序列的贡献；count 在 count_pairs 中加权一次。
            old_pair = count_pairs({token: count})
            new_pair = count_pairs({merged_tokens: count})
            pair_diff(old_pair, new_pair, pair_counts)
    return new_word_counts


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

if __name__ == "__main__":
    print("测试四：临对频率")
    train_bpe("toy_pair_count.txt", 300, ["<|endoftext|>"])
