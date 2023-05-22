from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = AutoModel.from_pretrained('bert-base-chinese')

# 分词并编码
token = tokenizer.encode('北京欢迎你')
print(token)
# [101, 1266, 776, 3614, 6816, 872, 102]
exit()

# 简写形式
token = tokenizer('北京欢迎你')

# 解码
print(tokenizer.decode([101, 1266, 776, 3614, 6816, 872, 102]))

# 查看特殊标记
print(tokenizer.special_tokens_map)

# 查看特殊标记对应id
print(tokenizer.encode(['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'], add_special_tokens=False))
# [100, 102, 0, 101, 103]