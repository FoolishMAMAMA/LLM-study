from minbpe import BasicTokenizer

tokenizer = BasicTokenizer()


tokenizer.train(
    text="abab",
    vocab_size=258,
    verbose =True
)


print("merges:", tokenizer.merges)
print(
    "new vocab:",
    {idx: token for idx, token in tokenizer.vocab.items() if idx >= 256}
)

# 2. 编码
ids = tokenizer.encode("ababab")
print("encoded:", ids)

# 3. 解码
text = tokenizer.decode(ids)
print("decoded:", text)