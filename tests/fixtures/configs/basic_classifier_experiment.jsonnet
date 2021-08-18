local embedding_dim = std.parseJson(std.extVar('embedding_dim'));

{
  dataset_reader: {
    type: 'text_classification_json',
    tokenizer: {
      type: 'whitespace',
    },
    token_indexers: {
      tokens: {
        type: 'single_id',
        namespace: 'tokens',
        lowercase_tokens: true,
      },
    },
    max_sequence_length: 400,
  },
  train_data_path: 'tests/fixtures/data/imdb_corpus.jsonl',
  validation_data_path: 'tests/fixtures/data/imdb_corpus.jsonl',
  model: {
    type: 'basic_classifier',
    text_field_embedder: {
      token_embedders: {
        tokens: {
          type: 'embedding',
          embedding_dim: embedding_dim,
          trainable: true,
        },
      },
    },
    seq2vec_encoder: {
      type: 'cnn',
      num_filters: 8,
      embedding_dim: embedding_dim,
      output_dim: 16,
    },
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 5,
    },
  },
  trainer: {
    checkpointer: 'mlflow',
    callbacks: ['mlflow'],
    optimizer: {
      type: 'adam',
      lr: 0.001,
    },
    validation_metric: '+accuracy',
    num_epochs: 3,
    grad_norm: 10.0,
    patience: 5,
    cuda_device: -1,
  },
}
