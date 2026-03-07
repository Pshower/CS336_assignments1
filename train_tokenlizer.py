import time
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import train_bpe


if __name__ == "__main__":

    GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 5000
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    # 保存测试结果
    # 写入词汇表 vocab_test.tsv (格式: id\tbytes_repr)
    with open("data/vocab_TinyStoriesV2_vocab.tsv", "w+", encoding="utf-8") as f:
        lines = [f"{idx}\t{token_bytes!r}\n" for idx, token_bytes in vocab.items()]
        f.writelines(lines)

    # 写入合并规则 merges_test.txt (格式: bytes_repr1\tbytes_repr2\n)
    with open("data/vocab_TinyStoriesV2_merges.txt", "w+", encoding="utf-8") as f:
        lines = [f"{a!r}\t{b!r}\n" for a, b in merges]
        f.writelines(lines)
    
    