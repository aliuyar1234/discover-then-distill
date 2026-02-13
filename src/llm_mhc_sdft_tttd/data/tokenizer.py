from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import sentencepiece as spm


@dataclass
class SpmTokenizer:
    """
    Thin wrapper around SentencePiece.
    """
    model_path: str

    def __post_init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model_path)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.sp.bos_id()] + ids
        if add_eos:
            ids = ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode(ids)

    def bos_id(self) -> int:
        return self.sp.bos_id()

    def eos_id(self) -> int:
        return self.sp.eos_id()

    def pad_id(self) -> int:
        return self.sp.pad_id()

    def unk_id(self) -> int:
        return self.sp.unk_id()
