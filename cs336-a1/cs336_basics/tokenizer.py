from cs336_basics.bpe_train import merge_pair
import regex as re
from cs336_basics.bpe_pretokenization import PAT
from collections.abc import Iterable, Iterator
import pickle


class Tokenizer:
    def __init__(self,vocab:dict[int, bytes],merges:list[tuple[bytes, bytes]],special_tokens: list[str] | None = None):
        self.vocab = dict(vocab)#这样分词器拥有自己的词表副本，之后添加特殊 token 时不会修改调用者的原字典。
        self.merges = merges
        if special_tokens:
            self.special_tokens = list(special_tokens) 
            self.special_token_set = set(self.special_tokens)
        else: 
            self.special_tokens=[]
            self.special_token_set = set()

        self.bytes_to_id ={}
        for token_id,token_bytes in self.vocab.items():
            self.bytes_to_id[token_bytes]=token_id

        

        for i in self.special_tokens:
            encode_res=i.encode("utf-8")
            if encode_res not in self.bytes_to_id:
                max_val = max(self.vocab.keys(), default=-1)
                self.vocab[max_val+1] = encode_res
                self.bytes_to_id[encode_res] = max_val+1

        #合并排名表：用于选择当前应优先应用的合并规则
        self.merge_ranks={}
        for rank,pair in enumerate(self.merges):
            self.merge_ranks[pair]=rank
    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_piece(self,piece:str) -> list[int]:
        text_bytes = piece.encode("utf8")
        tokens = tuple(bytes([b]) for b in text_bytes)
        missing_rank = float("inf")
        while len(tokens)>=2:
            best_rank = missing_rank
            for i in range(len(tokens)-1):
                pair = (tokens[i],tokens[i+1])
                rank = self.merge_ranks.get(pair,float("inf"))
                best_rank = min(rank,best_rank)
            if best_rank ==float("inf"):
                break #no thing can be merged
            tokens = merge_pair(tokens,self.merges[best_rank])
        int_result=list()
        for i in range(len(tokens)):
            int_result.append(self.bytes_to_id[tokens[i]])
        return int_result

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            pattern = "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))
            segments = [s for s in re.split(f"({pattern})", text) if s]
        else:
            segments = [text]
        res = []
        for segment in segments:
            if  segment in self.special_token_set:
                res.append(self.bytes_to_id[segment.encode("utf-8")])
            else:
                for match in re.finditer(PAT, segment):
                    res.extend(self._encode_piece(match.group()))
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            ids = self.encode(text)
            for token_id in ids:
                yield token_id

    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens: list[str] | None = None ):
        with open(vocab_filepath,"rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath,"rb") as g:
            merges = pickle.load(g)

        return cls(vocab,merges,special_tokens)
        