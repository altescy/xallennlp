from xallennlp.data.tokenizers import FugashiTokenizer


def test_fugashi_tokenizer() -> None:
    tokenizer = FugashiTokenizer()
    text = "これは例文です"
    tokens = [x.text for x in tokenizer.tokenize(text)]
    desired = ["これ", "は", "例文", "です"]

    assert tokens == desired
