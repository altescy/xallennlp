from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.data.fields import TextField
from xallennlp.data.dataset_readers import PreprocessReader
from xallennlp.data.preprocessors import Lowercase


def test_preprocess_reader_text_to_instance() -> None:
    reader = PreprocessReader(
        TextClassificationJsonReader(),
        {"text": Lowercase()},
    )
    instance = reader.text_to_instance(text="THIS IS A TEST SENTENCE")
    text_field = instance["tokens"]
    assert isinstance(text_field, TextField)

    desired_output = ["this", "is", "a", "test", "sentence"]
    assert [token.text for token in text_field.tokens] == desired_output
