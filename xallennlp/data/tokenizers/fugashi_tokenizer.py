from typing import List

from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer


@Tokenizer.register("fugashi")
class FugashiTokenizer(Tokenizer):
    def __init__(
        self,
    ) -> None:
        from fugashi.fugashi import Tagger

        super().__init__()

        self._tokenizer = Tagger()

    def tokenize(self, text: str) -> List[Token]:
        return [
            Token(
                text=fugashi_token.surface,
                lemma_=fugashi_token.feature.orthBase,
                pos_=fugashi_token.feature.pos1,
            )
            for fugashi_token in self._tokenizer(text)
        ]
