from typing import List

from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from fugashi.fugashi import Node, Tagger


@Tokenizer.register("fugashi")
class FugashiTokenizer(Tokenizer):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self._tokenizer = Tagger()

    def tokenize(self, text: str) -> List[Token]:
        return [self._convert_token(fugashi_token) for fugashi_token in self._tokenizer(text)]

    def _convert_token(self, fugashi_token: Node) -> Token:
        token = Token(
            text=fugashi_token.surface,
            lemma_=fugashi_token.feature.orthBase,
            pos_=fugashi_token.feature.pos1,
        )
        return token
