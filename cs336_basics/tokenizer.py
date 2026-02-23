import ast
from io import BytesIO
from typing import Iterable
from multiprocess import Pool

from pretokenization_example import find_chunk_boundaries
from train_bpe import process_chunk, NUM_THREAD


class Tokenizer:
    def  __init__(self, vocab: dict[int, bytes],
                  merges: list[tuple[bytes, bytes]],
                  special_tokens: list[str] | None=None):
        self.vocab = vocab
        self.vocab_reverse = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens # 这里会有干扰
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None=None):
        vocab: dict[int, bytes] = {}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            for line in f:
                id_str, token_repr = line.strip().split("\t")
                token_str = ast.literal_eval(token_repr)
                vocab[int(id_str)] = token_str

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                a_repr, b_repr = line.split('\t')
                a = ast.literal_eval(a_repr)
                b = ast.literal_eval(b_repr)
                merges.append((a, b))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)


    def apply_merge_to_chunk(self, chunk: list[list[bytes]]) -> list[int]:
        tokens_id = []
        for tokens in chunk:
            # print(f"tokens==>{tokens}")
            for pair in self.merges:
                # print(f"pair==>{pair}")
                b1, b2 = pair
                new_token = b1 + b2
                new_tokens: list[bytes] = []
                i = 0
                while(i < len(tokens)):
                    if(i < len(tokens) - 1 and tokens[i] == b1 and tokens[i + 1] == b2):
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
                # print(f"tokens==>{tokens}")
            # print(f"self.vocab_reverse==>{self.vocab_reverse}")
            for token in tokens:
                # print(f"token==>{token}")
                tokens_id.append(self.vocab_reverse.get(token))
            # print(f"tokens_id==>{tokens_id}")
        return tokens_id


    def encode(self, text: str) -> list[int]:
        # 预分词并设置并行
        input_files = BytesIO(text.encode())
        boundaries = find_chunk_boundaries(input_files, NUM_THREAD, bytes(self.special_tokens[0].encode("utf-8")))

        # 将文本设置为list[lits[bytes]]形式
        task_args = [(input_files, start, end, self.special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
        with Pool(processes=NUM_THREAD) as pool:
            chunk_results = pool.map(process_chunk, task_args)
        # print(f"chunk_results==>{chunk_results}")
        # 合并merges当中的词
        with Pool(processes=NUM_THREAD) as pool:
            tokens_id = pool.map(self.apply_merge_to_chunk, chunk_results)
        # print(f"tokens_id==>{tokens_id}")
        tokens_id_ret: list[int] = []
        for tokens_id_i in tokens_id:
            tokens_id_ret.extend(tokens_id_i)
        
        return tokens_id_ret

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for line in iterable:
            tokens_id = self.encode(line)
            yield from tokens_id

    def decode(self, ids: list[int]) -> str:
        tokens = bytes()
        vocab_size = len(self.vocab)

        replacement_char = "\uFFFD"
        for idi in ids:
            if idi < vocab_size:
                token = self.vocab[idi]
            else:
                token = bytes(replacement_char, encoding='utf-8')
            tokens += token
        
        return tokens.decode(encoding="utf-8", errors="replace")



if __name__ == "__main__":

    special_tokens = ["<|endoftext|>"]
    tokk = Tokenizer.from_files("data/vocab_test.tsv", "data/merges_test.txt", special_tokens)
    print(tokk.encode(" said"))
    print(tokk.decode([486, 258]))

