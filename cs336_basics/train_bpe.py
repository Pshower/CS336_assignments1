from io import BytesIO
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))
import regex as re
from collections import defaultdict, Counter
from multiprocess import Pool

from pretokenization_example import find_chunk_boundaries


NUM_THREAD = 16

def process_chunk(args: tuple[BytesIO, int, int, list[str]]) -> list[list[bytes]]:
    
    input_file, start, end, special_tokens = args

    input_file.seek(start)
    chunk = input_file.read(end - start).decode("utf-8", errors="ignore")

    # 1 对special_tokens转义并且用|连接，然后对字段分词获得文档
    pattern = "|".join(re.escape(token) for token in special_tokens)
    documents = re.split(pattern, chunk)

    # 2 将所有documents转换为bytes
    pre_tokens_bytes = []
    GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for doc in documents:
        tokens = [match.group(0).encode("utf-8") for match in re.finditer(GPT2_SPLIT_PATTERN, doc)]
        for token in tokens:
            token_bytes = [bytes([b]) for b in token]
            pre_tokens_bytes.append(token_bytes)

    return pre_tokens_bytes
    

def count_pairs(args: tuple[dict[int, bytes], list[list[bytes]]]) -> dict[int, int]:
    """
    统计当前chunk中所有单词的相邻token对频率。
    args: (chunk,) 其中chunk是一个list[list[bytes]]，每个list[bytes]是一个单词的token序列。
    返回Counter，键为相邻对 (bytes, bytes)，值为该对出现的次数。
    """
    voc_lib, chunk = args  # 直接传入chunk，因为voc_lib不需要在这里使用
    counts = Counter()
    for tokens in chunk:
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            counts[pair] += 1
    return counts


def apply_merge_to_chunk(args: tuple[tuple[bytes], list[list[bytes]]]) -> list[list[bytes]]:
    """
    将chunk当中的best_pairs合并
    """
    best_pairs, chunk = args
    b1, b2 = best_pairs
    new_chunk = []
    for tokens in chunk:
        new_tokens = []
        i = 0
        n = len(tokens)
        while i < n:
            if i < n - 1 and tokens[i] == b1 and tokens[i + 1] == b2:
                new_tokens.append(b1 + b2)   # 合并为新 token
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        new_chunk.append(new_tokens)
    return new_chunk


def train_bpe(input_path: str,
              vocab_size: int,
              special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    bpe训练主代码
    """
    
    # 1 初始化词表，包括256个ascii码和special_tokens
    voc_lib = {i : bytes([i]) for i in range(256)}
    for sp_token in special_tokens:
        voc_lib[len(voc_lib)] = sp_token.encode("utf-8")

    # 2 预分词并设置并行
    # 先将文件内容读入内存，避免跨进程传递文件句柄问题
    with open(input_path, "rb") as f:
        file_content = f.read()
    
    input_file = BytesIO(file_content)
    boundaries = find_chunk_boundaries(input_file, NUM_THREAD, bytes(special_tokens[0].encode("utf-8")))
    task_args = [(BytesIO(file_content), start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    
    with Pool(processes=NUM_THREAD) as pool:
        chunk_results = pool.map(process_chunk, task_args)

    # 合并词表
    merges = []

    # 3 bpe训练
    while(len(voc_lib) < vocab_size):
        # 3.1 对每个chunk_results当中的内容进行统计
        count_pairs_args = [(voc_lib, chunk) for chunk in chunk_results]
        with Pool(processes=NUM_THREAD) as pool:
            counts_results = pool.map(count_pairs, count_pairs_args)

        # 合并统计的字典
        counts = Counter()
        for count_result in counts_results:
            counts.update(count_result)
        
        if not counts:
            break
        
        # 3.2 选择频率最高的一对，同频时取字典序大的
        # 选择字段序最大的
        best_pairs = max(counts.items(), key=lambda item: (item[1], item[0]))[0]

        # 3.3 合并结果：更新词表和merges
        # print(f"test_best_pairs====> {best_pairs}")
        b1, b2 = best_pairs
        new_token = b1 + b2
        voc_lib[len(voc_lib)] = new_token
        merges.append((b1, b2))

        # 3.4 更新chunk中的token
        update_args = [(best_pairs, chunk) for chunk in chunk_results]
        with Pool(processes=NUM_THREAD) as pool:
            new_chunks = pool.map(apply_merge_to_chunk, update_args)
        chunk_results = new_chunks
    
    return voc_lib, merges


if __name__ == "__main__":
    GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    input_path = "data/TinyStoriesV2-GPT4-test.txt"
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    # 保存测试结果
    # 写入词汇表 vocab_test.tsv (格式: id\tbytes_repr)
    with open("data/vocab_test.tsv", "w+", encoding="utf-8") as f:
        lines = [f"{idx}\t{token_bytes!r}\n" for idx, token_bytes in vocab.items()]
        f.writelines(lines)

    # 写入合并规则 merges_test.txt (格式: bytes_repr1\tbytes_repr2\n)
    with open("data/merges_test.txt", "w+", encoding="utf-8") as f:
        lines = [f"{a!r}\t{b!r}\n" for a, b in merges]
        f.writelines(lines)

    input_path = "data/owt_train_test.txt"

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    
    # 保存测试结果
    # 写入词汇表 vocab_test.tsv (格式: id\tbytes_repr)
    with open("data/vocab_owt_test.tsv", "w+", encoding="utf-8") as f:
        lines = [f"{idx}\t{token_bytes!r}\n" for idx, token_bytes in vocab.items()]
        f.writelines(lines)

    # 写入合并规则 merges_test.txt (格式: bytes_repr1\tbytes_repr2\n)
    with open("data/merges_owt_test.txt", "w+", encoding="utf-8") as f:
        lines = [f"{a!r}\t{b!r}\n" for a, b in merges]
        f.writelines(lines)

    



