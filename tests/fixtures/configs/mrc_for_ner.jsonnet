local embedding_dim = 10;

{
    "dataset_reader": {
        "lazy": false,
        "type": "mrc_for_ner",
        "context_tokenizer": "whitespace",
        "query_tokenizer": "spacy",
    },
    "train_data_path": "tests/fixtures/data/mrc_ner.jsonl",
    "validation_data_path": "tests/fixtures/data/mrc_ner.jsonl",
    "model": {
        "type": "mrc_for_ner",
        "text_field_embedder": {
            "token_embedders": {
                "token": {
                    "type": "embedding",
                    "embedding_dim": embedding_dim,
                    "trainable": true,
                }
            }
        },
        "context_layer": {
          "type": "pass_through",
          "input_dim": embedding_dim,
        },
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 2
        },
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "validation_metric": "+fscore",
        "num_epochs": 5,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": -1
    }
}
