class Config:

    #################### For BERT fine-tuning ####################
    # atis dataset
    atis_train_path = "data/atis/raw_data.pkl"
    atis_test_path = "data/atis/raw_data_test.pkl"
    atis_dic_path = "data/atis/intent2id.pkl"
    atis_model_path = "checkpoints/epoch-atis.pth"
    atis_embedding_path = "finetune_results/atis_embeddings_with_hidden.pth"

    # semantic parsing dataset
    se_path = "data/semantic/raw_data_se.pkl"
    se_dic_path = "data/semantic/intent2id_se.pkl"
    se_model_path = "checkpoints/epoch-se.pth"
    se_embedding_path = "finetune_results/se_embeddings_with_hidden.pth"

    # multiWOZ dataset
    woz_path = "data/MULTIWOZ2.1/dialogue_data.pkl"
    woz_dic_path = "data/MULTIWOZ2.1/intent2id.pkl"
    woz_dialogue_id_path = "data/MULTIWOZ2.1/dialogue_id.pkl"
    woz_model_path = "checkpoints/epoch-woz-17.pth" 
    woz_embedding_path ="finetune_results/woz_embeddings.pth"

    maxlen = 20 #20
    batch_size = 128 #16
    epochs = 10 #5
    learning_rate_bert = 2e-5
    learning_rate_classifier = 1e-3

    # control
    datatype = "semantic"
    mode = "embedding" #"user", "data"

    #################### For Clustering & DCEC ####################
    
    dic_path = "/nethome/twu367/Multi-intent-dialoguer/data/semantic/intent2id_se.pkl"
    embedding_path = "/nethome/twu367/Multi-intent-dialoguer/finetune_results/se_embeddings_with_hidden.pth"
    woz_dic_path = "/nethome/twu367/Multi-intent-dialoguer/data/MULTIWOZ2.1/intent2id.pkl"
    woz_embedding_path = "/nethome/twu367/Multi-intent-dialoguer/finetune_results/woz_embeddings_sub.pth"
    
    # Model
    input_shape = (20, 768)
    filters = [16, 8, 1]
    kernel_size = 3
    alpha = 1

    # Training
    b_size = 1024
    n_clusters = 180 #180
    max_iter = 100
    update_interval = 10
    save_interval = 10
    tol = 1e-3

    weights = None #'checkpoints-dcec/dcec_model_att_99.h5'

    # clustering
    cluster_data_path = "clustering_results/data_att.pkl"
    cluster_label_path =  "clustering_results/labels_att.pkl"
    cluster_weight_path =  "clustering_results/weight_att.pkl"
    cluster_id = 0

    #################### For scBERT ####################
    se_path_for_sc = "data/semantic/raw_data_se_not_tokenize.pkl"
    se_dic_path_for_sc = "data/semantic/intent2id_se_not_tokenize.pkl"

    atis_path_for_sc = "data/atis/raw_data_not_tokenize.pkl"
    atis_dic_path_for_sc = "data/atis/intent2id_not_tokenize.pkl"

    neg_size = 100
    hidden_dim = 768





opt = Config()