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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from numpy import interp
from itertools import cycle
import numpy
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import keras 
from skmultilearn.model_selection import IterativeStratification
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import json


def getBidirectional(x, mode):
    main_lstm = Bidirectional(LSTM(units=80, return_sequences=True, recurrent_dropout=0.4), merge_mode=mode)(x)
    return main_lstm


def get_lstm_model(x, backwards):
    main_lstm = GRU(units=80, return_sequences=True, go_backwards=backwards, recurrent_dropout=0.4)(x)
    return main_lstm

def pred2label(pred):
    out = []
    #print(pred)
    for pred_i in pred:
    #    print(pred_i)
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    #print(out)
    return out
            
def idtotag(pred):
    out = []
    #print(pred)
    for pred_i in pred:
    #    print(pred_i)
        out_i = []
        for p in pred_i:
            p_i = idx2tag[p].replace("PAD", "O")
            out_i.append(p_i)
        out.append(out_i)
 #   print(out)
    return out

def readFile(filename):
    with open(filename) as f:
        content = f.readlines()
    return content
    
class SentenceGetter(object):    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

   

#data = pd.read_csv("small_whole_ndeath+ninjured.csv", encoding="latin1")
#loading labelled data
data = pd.read_csv("small.csv", encoding="latin1")
#loading test data that is unlabelled
realdata = pd.read_csv("2.csv", encoding="latin1")
data = data.fillna(method="ffill")
realdata = realdata.fillna(method="ffill")
train = True
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
data.tail(10)

words = list((data["Word"].values))
x = np.array(words) 
words=np.unique(x)
n_words = len(words); 
print('number of words '+str(n_words))

tags = list(data["Tag"].values)
x = np.array(tags) 
tags=np.unique(x)
n_tags = len(tags);
print('number of tags'+str(n_tags))



getter = SentenceGetter(data)
sent = getter.get_next()
sentences = getter.sentences
getter = SentenceGetter(realdata)
realsentences = getter.sentences
max_len = 75
max_len_char = 10
word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["UNK"] = 1
word2idx["PAD"] = 0
idx2word = {i: w for w, i in word2idx.items()}
tag2idx = {t: i + 1 for i, t in enumerate(tags)}
tag2idx["PAD"] = 0
idx2tag = {i: w for w, i in tag2idx.items()}
X_word = [[word2idx[w[0]] for w in s] for s in sentences]
realX_word = []
for s in realsentences:
    seq = []
    for w in s:
        if w[0] in word2idx.keys():
         seq.append( word2idx[w[0]])
        else:
            seq.append(word2idx["UNK"])
    realX_word.append (seq)
X_word = pad_sequences(maxlen=max_len, sequences=X_word, value=word2idx["PAD"], padding='post', truncating='post')
realX_word = pad_sequences(maxlen=max_len, sequences=realX_word, value=word2idx["PAD"], padding='post', truncating='post')
chars = set([w_i for w in words for w_i in w])
n_chars = len(chars)
char2idx = {c: i + 2 for i, c in enumerate(chars)}
char2idx["UNK"] = 1
char2idx["PAD"] = 0
X_char = []
for sentence in sentences:
    sent_seq = []
    for i in range(max_len):
        word_seq = []
        for j in range(max_len_char):
            try:
                word_seq.append(char2idx.get(sentence[i][0][j]))
            except:
                word_seq.append(char2idx.get("PAD"))
        sent_seq.append(word_seq)
    X_char.append(np.array(sent_seq))

realX_char = []
for sentence in realsentences:
    sent_seq = []
    for i in range(max_len):
        word_seq = []
        for j in range(max_len_char):
            try:
                word_seq.append(char2idx.get(sentence[i][0][j]))
            except:
                word_seq.append(char2idx.get("PAD"))
        sent_seq.append(word_seq)
    realX_char.append(np.array(sent_seq))
       
   
y = [[tag2idx[w[2]] for w in s] for s in sentences]
real_y = [[tag2idx[w[2]] for w in s] for s in realsentences]

y = pad_sequences(maxlen=max_len, sequences=y, value=tag2idx["PAD"], padding='post', truncating='post')
print("k fold verification")
if train == True: 
    #kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    kfold = IterativeStratification(n_splits=5, order=1)
    cvscores = []
    
    i=0
    for train, test in kfold.split(X_word, y):
        from sklearn.model_selection import train_test_split
        X_word_tr = X_word[train]
        
        X_word_te = X_word[test]
        y_tr = y[train]
        y_te = y[test]
        #print(y_te)
        X_char_te = np.array(X_char)[test.astype(int)]
        X_char_tr = np.array(X_char)[train.astype(int)]
        #X_char_te = X_char[test]
        word_in = Input(shape=(max_len,))
        emb_word = Embedding(input_dim=n_words + 2, output_dim=20,
                             input_length=max_len, mask_zero=True)(word_in)

        # input and embeddings for characters
        char_in = Input(shape=(max_len, max_len_char,))
        emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=10,
                                   input_length=max_len_char, mask_zero=True))(char_in)
        # character LSTM to get word encodings by characters
        char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                        recurrent_dropout=0.5))(emb_char)

        # main LSTM
        x = concatenate([emb_word, char_enc])
        x = SpatialDropout1D(0.3)(x)
        main_blstm = getBidirectional(x, 'sum')
        #main_blstm = get_lstm_model(x, False)
        blstm_out = TimeDistributed(Dense(n_tags + 1, activation="softmax"))(main_blstm)
        model_bidirectional = Model([word_in, char_in], blstm_out)
        model_bidirectional.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
        #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc = ModelCheckpoint('best_blstm_model.h5', monitor='val_acc', mode='max', verbose=0, save_best_only=True)
        #model_bidirectional.summary()
        print("training")
        blstm_history = model_bidirectional.fit([X_word_tr,
                             np.array(X_char_tr).reshape((len(X_char_tr), max_len, max_len_char))],
                            np.array(y_tr).reshape(len(y_tr), max_len, 1),
                            batch_size=32, epochs=100, validation_split=0.2, verbose=0, callbacks=[mc])

        
        
        
        hist = pd.DataFrame(blstm_history.history)
        hist_json_file = 'history'+str(i)+'.json' 
        with open(hist_json_file, mode='w') as f:
            hist.to_json(f)
        i=i+1
        loaded_model = load_model('best_blstm_model.h5')


        #y_pred = loaded_model.predict([realX_word,
        #                        np.array(realX_char).reshape((len(realX_char),
        #                                                     max_len, max_len_char))])
        
        train_acc = loaded_model.evaluate([X_word_tr,
                                np.array(X_char_tr).reshape((len(X_char_tr), max_len, max_len_char))],
                                np.array(y_tr).reshape(len(y_tr), max_len, 1), verbose=0)
        test_acc = loaded_model.evaluate([X_word_te,
                                np.array(X_char_te).reshape((len(X_char_te), max_len, max_len_char))],
                                np.array(y_te).reshape(len(y_te), max_len, 1), verbose=0)
        print(train_acc)
        print(test_acc)
        
        print (loaded_model.metrics_names)
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], test_acc[1]*100))
        cvscores.append(test_acc[1] * 100)
        
        
        
        
        test_pred = loaded_model.predict([X_word_te,
                                np.array(X_char_te).reshape((len(X_char_te), max_len, max_len_char))])
        
        idx2tag = {i: w for w, i in tag2idx.items()}
        np.set_printoptions(threshold=np.inf)
        
        
        pred_labels = pred2label(test_pred)
        test_labels = idtotag(y_te)

        

        #print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
        #classification_report = classification_report(test_labels, pred_labels)
        #print(classification_report)



       
        test_pred2 = loaded_model.predict([X_word_te,
                                np.array(X_char_te).reshape((len(X_char_te), max_len, max_len_char))])[:,1]
        mlb = MultiLabelBinarizer()
        y_enc = mlb.fit_transform(y_te)
        a = np.zeros((len(y_enc),  len(test_pred2[0])-len(y_enc[0])))
        
        
        y_enc = np.concatenate((y_enc, a), 1)
        #print((test_pred2.shape))
        #print((y_enc.shape))
        #print((y_te.shape))
        
        #fpr, tpr, threshold = roc_curve(y_enc, test_pred2)
        #roc_auc = auc(fpr, tpr)
        #print(roc_auc)

        
       
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for j in range(n_tags):
            fpr[j], tpr[j], _ = roc_curve(y_enc[:, j], test_pred2[:, j])
            #assert ~np.any(np.isnan(fpr)), "found nan fpr"
            #assert ~np.any(np.isnan(tpr)), "found nan tpr"
            roc_auc[j] = auc(fpr[j], tpr[j])
            #print("fpr"+str(fpr[j]))
            #print("tpr"+str(tpr[j]))

        # Compute micro-average ROC curve and ROC area
        #fpr["micro"], tpr["micro"], _ = roc_curve(y_enc.ravel(), test_pred.ravel())
        #roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # method I: plt
        #plt.title('Receiver Operating Characteristic')
        #plt.plot(fpr[1], tpr[1], 'b', label = 'AUC = %0.2f' % roc_auc[1])
        #plt.plot(fpr[1], tpr[1], 'b', label = 'AUC ')
        #plt.legend(loc = 'lower right')
        #plt.plot([0, 1], [0, 1],'r--')
        #plt.xlim([0, 1])
        #plt.ylim([0, 1])
        #plt.ylabel('True Positive Rate')
        #plt.xlabel('False Positive Rate')
        #plt.show()
        
        fig = plt.figure()
        lw = 2
        plt.plot(fpr[1], tpr[1], color='darkorange',
            lw=lw, label='B-geo, ROC curve (area = %0.2f)' % roc_auc[1])
        plt.plot(fpr[2], tpr[2], color='blue',
            lw=lw, label='B-gpe, ROC curve (area = %0.2f)' % roc_auc[2])
        
        plt.plot(fpr[4], tpr[4], color='aqua',
            lw=lw, label='B-ndeath, ROC curve (area = %0.2f)' % roc_auc[4])
        plt.plot(fpr[5], tpr[5], color='yellow',
            lw=lw, label='B-ninjured, ROC curve (area = %0.2f)' % roc_auc[5])
        plt.plot(fpr[6], tpr[6], color='green',
            lw=lw, label='B-org, ROC curve (area = %0.2f)' % roc_auc[6])
        plt.plot(fpr[7], tpr[7], color='red',
            lw=lw, label='B-per, ROC curve (area = %0.2f)' % roc_auc[7])
        plt.plot(fpr[8], tpr[8], color='cornflowerblue',
            lw=lw, label='B-time, ROC curve (area = %0.2f)' % roc_auc[8])
            
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        fig.savefig('kfold_run'+str(i)+'.png')
        
        
        all_fpr = np.unique(np.concatenate([fpr[k] for k in range(n_tags)]))
        #print(all_fpr)
            # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        #print(mean_tpr)
        for j in range(n_tags):
            mean_tpr += interp(all_fpr, fpr[j], tpr[j])
        #print(mean_tpr)
        # Finally average it and compute AUC
        mean_tpr /= n_tags
        #print("mean tpr"+str(mean_tpr))
        #print("all fpr"+str(all_fpr))
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        fig = plt.figure()
        #plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]),color='deeppink', linestyle=':', linewidth=4)
        lw = 2
        #plt.plot(fpr["macro"], tpr["macro"],
        #label='macro-average ROC curve (area = {0:0.2f})'
        #      ''.format(roc_auc["macro"]),
        #    color='navy', linestyle=':', linewidth=4)

        

        
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        
        
        
        
        
        #colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'yellow', 'green', 'red', 'blue', 'orange'])
        #for j, color in zip(range(n_tags), colors):
        #    plt.plot(fpr[j], tpr[j], color=color, lw=lw,
        #     label='ROC curve of class {0} (area = {1:0.2f})'
        #     ''.format(j, roc_auc[j]))

                
        #plt.show()
        


        
               


    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
     
 
hist = json.load(open('history0.json', 'r'))
df = pd.DataFrame(hist)
plt.style.use("ggplot")
fig = plt.figure(figsize=(12,12))
plt.plot(df["acc"])
plt.plot(df["val_acc"])
#plt.plot(flstmhist["acc"])
plt.legend(['train acc', 'validation acc'], loc='upper left')
fig.savefig('acc_res.png')
plt.show()

plt.style.use("ggplot")
fig = plt.figure(figsize=(12,12))
plt.plot(df["loss"])
plt.plot(df["val_loss"])
#plt.plot(flstmhist["acc"])
plt.legend(['train loss', 'validation loss'], loc='upper left')
fig.savefig('loss_res.png')
plt.show()

loaded_model = load_model('best_blstm_model.h5')


y_pred = loaded_model.predict([realX_word,
                        np.array(realX_char).reshape((len(realX_char),
                                                     max_len, max_len_char))])
                                                     
train_acc = loaded_model.evaluate([X_word_tr,
                        np.array(X_char_tr).reshape((len(X_char_tr), max_len, max_len_char))],
                        np.array(y_tr).reshape(len(y_tr), max_len, 1), verbose=0)
test_acc = loaded_model.evaluate([X_word_te,
                        np.array(X_char_te).reshape((len(X_char_te), max_len, max_len_char))],
                        np.array(y_te).reshape(len(y_te), max_len, 1), verbose=0)
print(train_acc)
print(test_acc)

#i = 0
#for res in y_pred:

    
#    p = np.argmax(res, axis=-1)
#    print("{:15}||{}".format("Word", "Pred"))
#    print(30 * "=")
#    for w, pred in zip(realX_word[i], p):
        
#       if w != 0:
#           print("{:15}:{}".format(idx2word[w], idx2tag[pred]))
         
#    i=i+1     


