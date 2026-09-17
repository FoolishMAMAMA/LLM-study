"""测试 BasicTokenizer 的 encode/decode 往返一致性"""
from minbpe.basic import BasicTokenizer

text = "你好，tokenizer 😊"
tokenizer = BasicTokenizer()

ids = tokenizer.encode(text)
print("encode 结果:", ids)

recovered = tokenizer.decode(ids)
print("decode 结果:", recovered)

assert recovered == text
print("✅ encode -> decode 往返一致")

# --- 训练后再验证一次 ---
train_text = "你好世界，你好tokenizer，你好😊" * 10
tokenizer.train(train_text, vocab_size=256 + 20, verbose=True)

ids = tokenizer.encode(text)
print("\n训练后 encode 结果:", ids)
assert tokenizer.decode(ids) == text
print("✅ 训练后往返一致")
