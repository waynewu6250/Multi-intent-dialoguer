from keras_bert import load_trained_model_from_checkpoint
from keras_bert import get_pretrained, PretrainedList, get_checkpoint_paths
from keras_bert import Tokenizer

from keras.layers import Input, Conv1D, MaxPool1D, Flatten, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model, load_model
import keras.backend as K
from keras.engine.topology import Layer

class MyTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            else:
                R.append('[UNK]')
        return R

class Attention(Layer):
    pass

def SCBert(opt):

    # Load bert model
    model_path = get_pretrained(PretrainedList.multi_cased_base)
    paths = get_checkpoint_paths(model_path)

    bert_model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, seq_len=opt.maxlen)

    for l in bert_model.layers:
        l.trainable = True

    











