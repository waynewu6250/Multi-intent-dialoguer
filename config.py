class Config:

    train_path = "data/raw_data.pkl" #"Multi-intent-dialoguer/data/raw_data.pkl"
    test_path = "data/raw_data_test.pkl"
    dic_path = "data/intent2id.pkl"
    model_path = "checkpoints/epoch-3.pth"

    maxlen = 20
    batch_size = 16
    epochs = 5
    learning_rate_bert = 2e-5
    learning_rate_classifier = 1e-3




opt = Config()