"""
NLP-Based Post-Processing - Step 7 of the Pipeline

- Spell correction
- Grammar / context-aware word correction
- Optional: BERT or language model for better corrections

Purpose: Improve readability and meaning of the decoded text.
"""

from __future__ import absolute_import, division, print_function
import re
from typing import List, Optional, Dict, Any

# Optional: symspellpy for fast spell correction
try:
    from symspellpy import SymSpell
    HAS_SYMSPELL = True
except ImportError:
    HAS_SYMSPELL = False

# Optional: BERT or transformers for context-aware correction
try:
    import torch
    from transformers import BertForMaskedLM, BertTokenizer
    HAS_BERT = True
except ImportError:
    HAS_BERT = False


class SpellCorrector:
    """
    Spell correction and optional BERT-based context correction.
    """

    def __init__(
        self,
        use_symspell: bool = True,
        use_bert: bool = False,
        symspell_max_edit: int = 2,
        symspell_prefix_length: int = 7,
        bert_model: str = "bert-base-uncased",
        custom_dict_path: Optional[str] = None,
    ):
        self.use_symspell = use_symspell and HAS_SYMSPELL
        self.use_bert = use_bert and HAS_BERT
        self._sym = None
        self._bert_model = None
        self._bert_tokenizer = None

        if self.use_symspell:
            self._sym = SymSpell(max_dictionary_edit_distance=symspell_max_edit, prefix_length=symspell_prefix_length)
            if custom_dict_path:
                self._sym.load_dictionary(custom_dict_path, term_index=0, count_index=1)
            else:
                # Use a small built-in vocabulary; user can load a full dictionary
                self._load_default_vocab()

        if self.use_bert:
            self._bert_model = BertForMaskedLM.from_pretrained(bert_model)
            self._bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
            self._bert_model.eval()

    def _load_default_vocab(self):
        # Minimal in-memory vocabulary for demo; replace with a real dictionary file in production
        common = [
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
            "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
            "how", "man", "new", "now", "old", "see", "way", "who", "boy", "did",
            "its", "let", "put", "say", "she", "too", "use", "that", "with", "have",
            "this", "will", "your", "from", "they", "been", "more", "were", "when",
            "what", "then", "them", "some", "into", "only", "other", "about", "their",
            "would", "there", "could", "other", "these", "first", "which", "where",
        ]
        for i, w in enumerate(common):
            self._sym.create_dictionary_entry(w, i + 1)

    def correct_word_symspell(self, word: str) -> str:
        if not self._sym or not word or not word.isalpha():
            return word
        suggestions = self._sym.lookup(word, max_edit_distance=2, verbosity=1)
        if suggestions:
            return suggestions[0].term
        return word

    def correct_bert_masked(self, text: str, top_k: int = 5) -> str:
        """Optional: replace a typo with [MASK], run BERT, pick best token. Simplified."""
        if not self.use_bert or not text.strip():
            return text
        # Simple heuristic: mask the shortest word that looks like a typo (e.g. len<=3 or has repeated chars)
        words = text.split()
        for i, w in enumerate(words):
            if len(w) <= 2 or not w.isalpha():
                continue
            masked = words[:i] + ["[MASK]"] + words[i + 1:]
            sent = " ".join(masked)
            # Run BERT (Pytorch)
            inp = self._bert_tokenizer(sent, return_tensors="pt")
            with torch.no_grad():
                logits = self._bert_model(**inp).logits
            idx = self._bert_tokenizer.convert_tokens_to_ids("[MASK]")
            mask_pos = (inp["input_ids"] == self._bert_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            if len(mask_pos) == 0:
                continue
            scores = logits[0, mask_pos[0], :]
            pred = scores.argmax().item()
            tok = self._bert_tokenizer.convert_ids_to_tokens(pred)
            if tok and not tok.startswith("["):
                words[i] = tok
                return " ".join(words)
        return text

    def correct(self, text: str, use_bert: Optional[bool] = None) -> str:
        if not text or not text.strip():
            return text
        use_bert = use_bert if use_bert is not None else self.use_bert

        # Optional: split on punctuation to preserve it
        parts = re.split(r'(\s+)', text)
        out = []
        for p in parts:
            if p.isspace() or not p.strip():
                out.append(p)
                continue
            word = p.strip()
            if word and word.isalpha():
                word = self.correct_word_symspell(word)
            out.append(word)
        text = "".join(out)

        if use_bert and self.use_bert:
            text = self.correct_bert_masked(text)
        return text


def correct_text(
    text: str,
    use_symspell: bool = True,
    use_bert: bool = False,
    **kwargs: Any,
) -> str:
    """One-shot correction."""
    c = SpellCorrector(use_symspell=use_symspell, use_bert=use_bert, **kwargs)
    return c.correct(text)


def load_corrector(
    use_symspell: bool = True,
    use_bert: bool = False,
    custom_dict_path: Optional[str] = None,
    **kwargs: Any,
) -> SpellCorrector:
    return SpellCorrector(
        use_symspell=use_symspell,
        use_bert=use_bert,
        custom_dict_path=custom_dict_path,
        **kwargs,
    )
