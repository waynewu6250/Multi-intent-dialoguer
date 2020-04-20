from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Lambda
from keras.models import Model, load_model
from keras import optimizers

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from model.cae import CAE_model, CustomLoss

class ClusterLayer(Layer):
    """
    N: Number of instances
    D: feature dimension
    C: Cluster number
    Input: features | shape: (N, D)
    Ouptut: Probability of features belonging each cluster | shape: (N, C)
    Weights: Cluster Centers | shape: (C, D)
    """

    def __init__(self, n_clusters, alpha = 1.0, **kwargs):
        
        super(ClusterLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
    
    def build(self, input_shape):

        D = input_shape[1]
        
        self.input_spec = InputSpec(ndim=2, dtype=K.floatx(), shape=(None, D))
        self.clusters = self.add_weight(shape=(self.n_clusters, D), initializer='glorot_uniform', name='clusters')
        self.built = True
    
    def call(self, inputs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_clusters)
    
    def get_config(self):
        my_config = {"n_cluster": self.n_clusters}
        base_config = super(ClusterLayer, self).get_config()
        return dict(list(base_config.items())+list(my_config.items()))


class DCEC:

    def __init__(self, input_shape, filters, kernel_size, n_clusters, weights, data, alpha=1.0, pretrain=True):
        
        if pretrain:
            self.autoencoder = CAE_model(input_shape, filters, kernel_size)
            self.autoencoder.load_weights('checkpoints-dcec/model_att.h5')
        else:
            print("Start Pretraining...")
            self.autoencoder = CAE_model(input_shape, filters, kernel_size)
            (emb_train, emb_test, att_train, att_test, l_train, l_test) = data
            self.pretrain_model(att_train, l_train)
            print("Pretraining Complete")
        
        features = self.autoencoder.get_layer('max_pooling1d_3').output
        
        self.feature_extractor = Model(self.autoencoder.input, features)
        
        features_s = Lambda(lambda x: K.squeeze(x, axis=2))(features)
        probs = ClusterLayer(n_clusters, alpha, name='cluster')(features_s)
        loss = CustomLoss()([self.autoencoder.get_layer('average_vector').output, self.autoencoder.output, self.autoencoder.get_layer('att_weights').output])
        
        self.model = Model(inputs = self.autoencoder.input, outputs=probs)

        if weights:
            self.model.load_weights(weights)
            print("2. Successfully loading weights!!")
    
    def pretrain_model(self, x_train, l_train):
        sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self.autoencoder.compile(optimizer=sgd)
        
        self.autoencoder.fit([x_train, l_train],
                epochs=30,
                batch_size=128,
                shuffle=True,
                validation_split=0.1)
        
        self.autoencoder.save_weights('checkpoints-dcec/model_woz.h5')
        self.pretrain = True
    
    @staticmethod
    def target(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T
    
    def compile(self, loss, optimizer):
        self.model.compile(loss=loss, optimizer=optimizer)

    def fit(self, data, opt):

        (emb_train, emb_test, att_train, att_test, l_train, l_test) = data
        
        # Initialize cluster centers by K-Means
        features = self.feature_extractor.predict([att_test, l_test])
        features = features.squeeze(-1)

        kmeans_model = KMeans(n_clusters=opt.n_clusters, n_init = 20, random_state=1)
        prev_label = kmeans_model.fit_predict(features)
        self.model.get_layer(name='cluster').set_weights([kmeans_model.cluster_centers_])

        # Start deep clustering training
        index = 0
        for iter in range(opt.max_iter):

            # Update our target distribution
            if iter % opt.update_interval == 0:

                q = self.model.predict([att_test, l_test])
                p = self.target(q)
                self.cur_label = np.argmax(q, axis = 1)

                # Check when to stop
                diff = np.sum(self.cur_label != prev_label).astype(np.float32) / self.cur_label.shape[0]
                prev_label = np.copy(self.cur_label)
                if iter > 0 and diff < opt.tol:
                    print('Difference ', diff, 'is smaller than tol ', opt.tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break
            
            # train on batch
            if (index + 1) * opt.batch_size > att_test.shape[0]:
                loss = self.model.train_on_batch(x=[att_test[index * opt.batch_size::], l_test[index * opt.batch_size::]],
                                                 y=p[index * opt.batch_size::])#, att_test[index * opt.batch_size::]])
                index = 0
            else:
                loss = self.model.train_on_batch(x=[att_test[index * opt.batch_size:(index + 1) * opt.batch_size], l_test[index * opt.batch_size:(index + 1) * opt.batch_size]],
                                                 y=p[index * opt.batch_size:(index + 1) * opt.batch_size])#,
                                                    #att_test[index * opt.batch_size:(index + 1) * opt.batch_size]])
                index += 1

            # save intermediate model
            if (iter+1) % opt.save_interval == 0:
                # save DCEC model checkpoints
                print('Saving model no.', iter)
                self.model.save_weights('checkpoints-dcec/dcec_model_woz_' + str(iter) + '.h5')
            

        
            


            








        


        
        



        
        









