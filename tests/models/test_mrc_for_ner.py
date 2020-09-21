from allennlp.common.testing import ModelTestCase

from xallennlp.models import MrcForNer


class TestMrcForNer(ModelTestCase):
    def setup(self):
        super().setup_method()
        self.config_file = "tests/fixtures/configs/mrc_for_ner.jsonnet"
        self.dataset_file = "tests/fixtures/data/mrc_ner.json"
        self.set_up_model(self.config_file, self.dataset_file)

    def test_mrc_for_ner_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
