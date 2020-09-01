{
    "dataset_reader": {
        "lazy": false,
        "type": "mrc_for_ner",
        "context_tokenizer": "whitespace",
        "query_tokenizer": "spacy",
    },
    "train_data_path": "data/mrc_ner.json",
    "validation_data_path": "data/mrc_ner.json",
    "model": {
        "type": "mrc_for_ner",
        "text_field_embedder": {
            "token_embedders": {
                "token": {
                    "type": "embedding",
                    "embedding_dim": 10,
                    "trainable": true,
                }
            }
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
        "validation_metric": "+f1",
        "num_epochs": 20,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": -1
    }
}
