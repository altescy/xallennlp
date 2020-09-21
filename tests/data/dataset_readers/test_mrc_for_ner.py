from xallennlp.data.dataset_readers import MrcForNerDatasetReader


class TestMrcForNerDatasetReader:
    def setup(self):
        self.context = "xinhua news agency , shanghai , august 31st , by reporter jierong zhou"
        self.query = (
            "geographical political entities are geographical regions defined by "
            "political and or social groups such as countries, nations, regions, "
            "cities, states, government and its people. ")
        self.metadata = {
            "start_position": [4],
            "end_position": [5],
            "span_position": ["4,5"],
        }
        self.reader = MrcForNerDatasetReader()

    def test_read(self):
        list(self.reader.read("tests/fixtures/data/mrc_ner.json"))

    def test_text_to_instance(self):
        instance = self.reader.text_to_instance(self.context, self.query,
                                                self.metadata)

        assert "text" in instance
        assert "span" in instance
        assert "start_position" in instance
        assert "end_position" in instance
        assert "span_position" in instance
        assert "metadata" in instance

    def test_span_position_to_array(self):
        array = self.reader._span_position_to_array(["2,3", "4,5"], length=8)  # pylint: disable=protected-access

        assert array.shape == (8, 8)
        assert array[0, 0] == 0
        assert array[2, 3] == 1
        assert array[4, 5] == 1
