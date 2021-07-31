import pandas as pd
import numpy as np
from keras.models import Model, Input
from keras.layers import LSTM, RNN, GRU, Embedding, Dense, TimeDistributed, Dropout, Conv1D
from keras.layers import Bidirectional, concatenate, SpatialDropout1D, GlobalMaxPooling1D
from keras.models import load_model
from collections import OrderedDict
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import StratifiedKFold





import numpy
# fix random seed for reproducibility



       
    
#import keras 
#from skmultilearn.model_selection import IterativeStratification
    
 
import json
import pandas as pd

import json
LSTM_hist = json.loads(open('history/fsltm_run_3/history0.json').read())
#LSTM_hist = pd.read_json('history/fsltm_run_3/history0.json', orient=1)
print (LSTM_hist["acc"])

for key in LSTM_hist["acc"].values():
    print(key)

#data = json.loads(s, object_pairs_hook=OrderedDict)


#data = json.load(open('history/fsltm_run_3/history0.json', 'r', ), object_pairs_hook=OrderedDict)
#LSTM_hist = pd.DataFrame.from_dict(data, orient=1)
#print (LSTM_hist.loc[[0:2],['acc','val_acc']])
#print (LSTM_hist.iloc[:,0])

#LSTM_df = pd.DataFrame(LSTM_hist)
BLSTM_hist = json.load(open('history/blstm_run2/history3.json', 'r'))
#BLSTM_df = pd.DataFrame(BLSTM_hist)

import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams.update({'font.size': 16})
f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(12, 8))
f.subplots_adjust(hspace=0.5, wspace=0.2)
f.tight_layout(pad=3.0)


ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')

ax2.set_xlabel('epoch')
ax2.set_ylabel('log loss')

ax1.set_title('model accuracy')
ax2.set_title('model loss')        

ax1.plot(list(LSTM_hist["acc"].values()), label="LSTM train acc")
ax1.plot(list(LSTM_hist["val_acc"].values()), label="LSTM validation acc")
ax1.plot(list(BLSTM_hist["acc"].values()), label="BLSTM train acc")
ax1.plot(list(BLSTM_hist["val_acc"].values()), label="BLSTM validation acc")
ax1.legend()

ax2.set_yscale('log')
ax2.plot(list(LSTM_hist["loss"].values()), label="LSTM train loss")
ax2.plot(list(LSTM_hist["val_loss"].values()), label="LSTM validation loss")
ax2.plot(list(BLSTM_hist["loss"].values()), label="BLSTM train loss")
ax2.plot(list(BLSTM_hist["val_loss"].values()), label="BLSTM validation loss")
ax2.legend()
        
#ax1=fig.add_subplot(2, 1, 1)
f.savefig('loss_res.png')
plt.show()

