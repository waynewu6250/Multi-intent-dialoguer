class Config:

    #################### For BERT fine-tuning ####################
    # control
    datatype = "e2e"
    mode = "embedding" #"user", "data"

    if datatype == "atis":
        # atis dataset
        train_path = "data/atis/raw_data.pkl"
        test_path = "data/atis/raw_data_test.pkl"
        dic_path = "data/atis/intent2id.pkl"
        model_path = "checkpoints/best_atis.pth"
        embedding_path = "finetune_results/atis_embeddings_with_hidden.pth"
    
    elif datatype == "semantic":
        # semantic parsing dataset
        train_path = "data/semantic/raw_data_se.pkl"
        test_path = None
        dic_path = "data/semantic/intent2id_se.pkl"
        model_path = None #"checkpoints/best_semantic.pth"
        embedding_path = "finetune_results/se_embeddings_with_hidden.pth"
    
    elif datatype == "e2e":
        # Microsoft e2e dialogue dataset
        train_path = "data/e2e_dialogue/dialogue_data.pkl"
        test_path = None
        dic_path = "data/e2e_dialogue/intent2id.pkl"
        model_path = None #"checkpoints/best_e2e.pth"
        embedding_path = "finetune_results/e2e_embeddings_with_hidden.pth"
    
    elif datatype == "woz":
        # multiWOZ dataset
        train_path = "data/MULTIWOZ2.1/dialogue_data.pkl"
        test_path = None
        dic_path = "data/MULTIWOZ2.1/intent2id.pkl"
        dialogue_id_path = "data/MULTIWOZ2.1/dialogue_id.pkl"
        model_path = "checkpoints/epoch-woz-17.pth" 
        embedding_path ="finetune_results/woz_embeddings_sub.pth"


    maxlen = 50 #20
    batch_size = 128 #16
    epochs = 20 #5
    learning_rate_bert = 2e-5
    learning_rate_classifier = 1e-3

    #################### For Clustering & DCEC ####################
    
    dic_path = "./data/semantic/intent2id_se.pkl"
    embedding_path = "./finetune_results/se_embeddings_raw_with_hidden.pth"
    woz_dic_path = "./data/MULTIWOZ2.1/intent2id.pkl"
    woz_embedding_path = "./finetune_results/woz_embeddings_sub.pth"
    
    # Model
    input_shape = (20, 768)
    filters = [16, 8, 1]
    kernel_size = 3
    alpha = 1

    # Training
    b_size = 1024
    n_clusters = 8 #180
    max_iter = 100
    update_interval = 10
    save_interval = 10
    tol = 1e-3

    weights = None #'checkpoints-dcec/dcec_model_att_99.h5'

    # clustering
    cluster_data_path = "clustering_results/data_att_woz_pair.pkl"
    cluster_label_path =  "clustering_results/labels_att_woz_pair.pkl"
    cluster_weight_path =  "clustering_results/weight_att_woz_pair.pkl"
    cluster_id = 0

    #################### For scBERT ####################

    se_path_for_sc = "data/semantic/raw_data_se_not_tokenize.pkl"
    se_dic_path_for_sc = "data/semantic/intent2id_se_not_tokenize.pkl"

    atis_path_for_sc = "data/atis/raw_data_not_tokenize.pkl"
    atis_dic_path_for_sc = "data/atis/intent2id_not_tokenize.pkl"

    neg_size = 100
    hidden_dim = 768





opt = Config()