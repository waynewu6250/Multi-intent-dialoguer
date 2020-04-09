class Config:

    #################### For BERT fine-tuning ####################
    # atis dataset
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

    # multiWOZ dataset
    woz_path = "data/"

    maxlen = 20
    batch_size = 16
    epochs = 5
    learning_rate_bert = 2e-5
    learning_rate_classifier = 1e-3

    # control
    datatype = "semantic"
    mode = "embedding" #"user", "data"

    #################### For Clustering ####################
    dic_path = "/nethome/twu367/Multi-intent-dialoguer/data/semantic/intent2id_se.pkl"
    embedding_path = "/nethome/twu367/Multi-intent-dialoguer/results/se_embeddings_raw.pth"

    #################### For DCEC ####################
    
    # Model
    input_shape = (768, 1)
    filters = [16, 8, 1]
    kernel_size = 3
    alpha = 1

    # Training
    b_size = 1024
    n_clusters = 180
    max_iter = 100
    update_interval = 10
    save_interval = 10
    tol = 1e-3

    weights = None #'checkpoints-dcec/dcec_model_99.h5'

    # clustering
    cluster_data_path = "clustering_results/data.pkl"
    cluster_label_path =  "clustering_results/labels.pkl"
    cluster_id = 0

    #################### For scBERT ####################
    se_path_for_sc = "data/semantic/raw_data_se_not_tokenize.pkl"
    se_dic_path_for_sc = "data/semantic/intent2id_se_not_tokenize.pkl"
    se_model_path_for_sc = "checkpoints-dcec/epoch-se.pth"

    neg_size = 100
    hidden_dim = 768





opt = Config()