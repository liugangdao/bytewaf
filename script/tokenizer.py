import tiktoken

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("cl100k_base")
    print(tokenizer.encode("hello world"))

    print(tokenizer._encode_bytes("hello world".encode()))
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")

    # 查看所有 special tokens（字符串）
    print(enc.special_tokens_set)

    # 查看 special tokens 及其对应的 ID
    print(enc._special_tokens)

    vocab_size = len(enc._mergeable_ranks) + len(enc._special_tokens)

    print(f"词表大小: {vocab_size}")  # 输出大约 100264
