class Config:

    # ATIS dataset
    atis_train_path = "data/atis/raw_data.pkl"
    atis_test_path = "data/atis/raw_data_test.pkl"
    atis_dic_path = "data/atis/intent2id.pkl"
    atis_model_path = "checkpoints/epoch-3.pth"
    atis_embedding_path = "results/atis_embeddings.pth"

    # semantic parsing dataset
    se_path = "data/semantic/raw_data_se.pkl"
    se_dic_path = "data/semantic/intent2id_se.pkl"
    se_model_path = "checkpoints/epoch-se-3.pth"
    se_embedding_path = "results/se_embeddings.pth"

    # model hyperparameter
    maxlen = 20
    batch_size = 16
    epochs = 5
    learning_rate_bert = 2e-5
    learning_rate_classifier = 1e-3

    # control
    datatype = "semantic"
    mode = "user" #"data"




opt = Config()