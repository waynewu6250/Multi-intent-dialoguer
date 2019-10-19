class Config:

    data_path = "data/raw_data.pkl" #"Multi-intent-dialoguer/data/raw_data.pkl"
    model_path = None #"bert_test.pth"

    maxlen = 256
    batch_size = 16
    epochs = 25
    learning_rate_bert = 1e-3
    learning_rate_classifier = 1e-3




opt = Config()