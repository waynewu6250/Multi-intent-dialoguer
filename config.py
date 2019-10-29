class Config:

    data_path = "data/raw_data.pkl" #"Multi-intent-dialoguer/data/raw_data.pkl"
    dic_path = "data/intent2id.pkl"
    model_path = None #"bert_test.pth"

    maxlen = 20
    batch_size = 16
    epochs = 25
    learning_rate_bert = 1e-3
    learning_rate_classifier = 1e-3




opt = Config()