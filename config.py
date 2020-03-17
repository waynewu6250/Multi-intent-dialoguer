class Config:

    #################### For BERT fine-tuning ####################
    atis_train_path = "data/atis/raw_data.pkl"
    atis_test_path = "data/atis/raw_data_test.pkl"
    atis_dic_path = "data/atis/intent2id.pkl"
    atis_model_path = "checkpoints/epoch-3.pth"
    atis_embedding_path = "results/atis_embeddings.pth"

    se_path = "data/semantic/raw_data_se.pkl"
    se_dic_path = "data/semantic/intent2id_se.pkl"
    se_model_path = "checkpoints/epoch-se-3.pth"
    se_embedding_path = "results/se_embeddings.pth"

    maxlen = 20
    batch_size = 16
    epochs = 5
    learning_rate_bert = 2e-5
    learning_rate_classifier = 1e-3

    datatype = "semantic"
    mode = "embedding" #"user", "data"

    #################### For Clustering ####################
    dic_path = "/nethome/twu367/Multi-intent-dialoguer/data/semantic/intent2id_se.pkl"
    embedding_path = "/nethome/twu367/Multi-intent-dialoguer/results/se_embeddings.pth"

    #################### For DCEC ####################
    
    # Model
    input_shape = (768, 1)
    filters = [16, 8, 1]
    kernel_size = 3
    alpha = 1

    # Training
    b_size = 1024
    n_clusters = 10
    max_iter = 100
    update_interval = 10
    save_interval = 10
    tol = 1e-3

    weights = None #'checkpoints/dcec_model_48.h5'

    cluster_id = 0



opt = Config()