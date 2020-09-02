from xallennlp.data.dataset_readers import SciERCReader


class TestSciERC:
    def setup(self):
        self.scierc_path = "tests/fixtures/data/scierc.jsonl"

    def test_read(self):
        reader = SciERCReader()
        instances = list(reader.read(self.scierc_path))

        assert len(instances) == 2

        num_spans = 0
        for sentence in instances[0]["metadata"]["original_sentences"]:
            num_tokens = len(sentence)
            num_spans += num_tokens * (num_tokens + 1) / 2
        assert len(instances[0]["spans"]) == num_spans
