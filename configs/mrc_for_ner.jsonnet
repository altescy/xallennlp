local embedding_dim = 768;

{
    "dataset_reader": {
        "lazy": false,
        "type": "mrc_for_ner",
        "context_tokenizer": "whitespace",
        "query_tokenizer": "spacy",
        "token_indexers": {
          "bert": {
            "type": "pretrained_transformer_mismatched",
            "max_length": 512,
            "model_name": "allenai/scibert_scivocab_cased"
          }
        },
    },
    "train_data_path": "https://s3.wasabisys.com/datasets-altescy/mrc_ner/ace04/train.jsonl",
    "validation_data_path": "https://s3.wasabisys.com/datasets-altescy/mrc_ner/ace04/dev.jsonl",
    "model": {
        "type": "mrc_for_ner",
        "text_field_embedder": {
          "token_embedders": {
            "bert": {
              "type": "pretrained_transformer_mismatched",
              "max_length": 512,
              "model_name": "allenai/scibert_scivocab_cased"
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
            "batch_size": 10
        },
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "validation_metric": "+fscore",
        "num_epochs": 10,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": 0
    }
}
