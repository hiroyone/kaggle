# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"_cell_guid": "f14a7be5-a88d-4f4d-95f8-fcfcafa39bc0", "_uuid": "c41191e2f86f00e8bdac86b3383b2d0bb43d247f", "cell_type": "markdown"}
# In this kernel, we shall see if pretrained embeddings like Word2Vec, GLOVE and Fasttext, which are pretrained using billions of words could improve our accuracy score as compared to training our own embedding. We will compare the performance of models using these pretrained embeddings against the baseline model that doesn't use any pretrained embeddings in my previous kernel [here](https://www.kaggle.com/sbongo/for-beginners-tackling-toxic-using-keras).
#
# ![](https://qph.fs.quoracdn.net/main-qimg-3e812fd164a08f5e4f195000fecf988f)
#
# Perhaps it's a good idea to briefly step in the world of word embeddings and see what's the difference between Word2Vec, GLOVE and Fasttext.
#
# Embeddings generally represent geometrical encodings of words based on how frequently appear together in a text corpus. Various implementations of word embeddings described below differs in the way as how they are constructed.
#
# **Word2Vec**
#
# The main idea behind it is that you train a model on the context on each word, so similar words will have similar numerical representations.
#
# Just like a normal feed-forward densely connected neural network(NN) where you have a set of independent variables and a target dependent variable that you are trying to predict, you first break your sentence into words(tokenize) and create a number of pairs of words, depending on the window size. So one of the combination could be a pair of words such as ('cat','purr'), where cat is the independent variable(X) and 'purr' is the target dependent variable(Y) we are aiming to predict.
#
# We feed the 'cat' into the NN through an embedding layer initialized with random weights, and pass it through the softmax layer with ultimate aim of predicting 'purr'. The optimization method such as SGD minimize the loss function "(target word | context words)" which seeks to minimize the loss of predicting the target words given the context words. If we do this with enough epochs, the weights in the embedding layer would eventually represent the vocabulary of word vectors, which is the "coordinates" of the words in this geometric vector space.
#
# ![](https://i.imgur.com/R8VLFs2.png)
#
# The above example assumes the skip-gram model. For the Continuous bag of words(CBOW), we would basically be predicting a word given the context. 
#
# **GLOVE**
#
# GLOVE works similarly as Word2Vec. While you can see above that Word2Vec is a "predictive" model that predicts context given word, GLOVE learns by constructing a co-occurrence matrix (words X context) that basically count how frequently a word appears in a context. Since it's going to be a gigantic matrix, we factorize this matrix to achieve a lower-dimension representation. There's a lot of details that goes in GLOVE but that's the rough idea.
#
# **FastText**
#
# FastText is quite different from the above 2 embeddings. While Word2Vec and GLOVE treats each word as the smallest unit to train on, FastText uses n-gram characters as the smallest unit. For example, the word vector ,"apple", could be broken down into separate word vectors units as "ap","app","ple". The biggest benefit of using FastText is that it generate better word embeddings for rare words, or even words not seen during training because the n-gram character vectors are shared with other words. This is something that Word2Vec and GLOVE cannot achieve.
#

# + {"_cell_guid": "bbf5b274-f78a-49c9-b416-d510adb2b399", "_uuid": "96724c148074572ca78eb67573b74bea3ea48266", "cell_type": "markdown"}
# Let's start off with the usual importing pandas, etc

# + {"_cell_guid": "b6103d10-10b9-44dd-8c7c-ce71bb390833", "_uuid": "908cc6a5d46bae29f822e0dd353fd34947898706"}
import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
import matplotlib.pyplot as plt
# %matplotlib inline

import gc
# -

from gensim.models import Word2Vec

# + {"_cell_guid": "38caf564-a3b9-455e-8764-be6af9fc9eb9", "_uuid": "95f368df56588643270c781668fa0a51b183a7a5", "cell_type": "markdown"}
# Some preprocessing steps that we have taken in my earlier kernel.

# + {"_cell_guid": "05c286a1-2510-4c38-8922-6f4d8dcad863", "_uuid": "bead9ea1dfe1846027bfcd94a264a110b6f06e81"}
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
embed_size=0

# + {"_cell_guid": "7e8ad169-36d2-4f13-8a6d-0b829451d6be", "_uuid": "a410746ad2dd36a2b9719382e04431eee5d34c68"}
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_train = train["comment_text"]
list_sentences_test = test["comment_text"]

# + {"_cell_guid": "3d2dad9a-7ce2-42a7-be74-c737618294aa", "_uuid": "a46ed61c7a46e6201d070531ab1794e998268c0e"}
max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

# + {"_cell_guid": "7d7a8d69-2ade-472b-8c6b-fdb6a916f24c", "_uuid": "b9e0c623fb56fa637df7a9f0659d36f9c639ac51"}
maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

# + {"_cell_guid": "dcd235bd-acf2-4767-9f4b-6244b92e6e54", "_uuid": "f9049f04bc2070efb6ee8cd8c764746807f3a7eb", "cell_type": "markdown"}
# Since we are going to evaluate a few word embeddings, let's define a function so that we can run our experiment properly. I'm going to put some comments in this function below for better intuitions.
#
# Note that there are quite a few GLOVE embeddings in Kaggle datasets, and I feel that it would be more applicable to use the one that was trained based on Twitter text. Since the comments in our dataset consists of casual, user-generated short message, the semantics used might be very similar. Hence, we might be able to capture the essence and use it to produce a good accurate score.
#
# Similarly, I have used the Word2Vec embeddings which has been trained using Google Negative News text corpus, hoping that it's negative words can work better in our "toxic" context.

# + {"_cell_guid": "c884b17c-3365-4f05-bad3-b90928e07719", "_uuid": "fc2ad52bb6b418694dca15370c6b9bc47b2e304b"}
def loadEmbeddingMatrix(typeToLoad):
        #load different embedding file from Kaggle depending on which embedding 
        #matrix we are going to experiment with
        if(typeToLoad=="glove"):
            EMBEDDING_FILE='../input/glove-twitter/glove.twitter.27B.25d.txt'
            embed_size = 25
        elif(typeToLoad=="word2vec"):
            word2vecDict = word2vec.KeyedVectors.load_word2vec_format("../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin", binary=True)
            embed_size = 300
        elif(typeToLoad=="fasttext"):
            EMBEDDING_FILE='../input/fasttext/wiki.simple.vec'
            embed_size = 300

        if(typeToLoad=="glove" or typeToLoad=="fasttext" ):
            embeddings_index = dict()
            #Transfer the embedding weights into a dictionary by iterating through every line of the file.
            f = open(EMBEDDING_FILE)
            for line in f:
                #split up line into an indexed array
                values = line.split()
                #first index is word
                word = values[0]
                #store the rest of the values in the array as a new array
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs #50 dimensions
            f.close()
            print('Loaded %s word vectors.' % len(embeddings_index))
        else:
            embeddings_index = dict()
            for word in word2vecDict.wv.vocab:
                embeddings_index[word] = word2vecDict.word_vec(word)
            print('Loaded %s word vectors.' % len(embeddings_index))
            
        gc.collect()
        #We get the mean and standard deviation of the embedding weights so that we could maintain the 
        #same statistics for the rest of our own random generated weights. 
        all_embs = np.stack(list(embeddings_index.values()))
        emb_mean,emb_std = all_embs.mean(), all_embs.std()
        
        nb_words = len(tokenizer.word_index)
        #We are going to set the embedding size to the pretrained dimension as we are replicating it.
        #the size will be Number of Words in Vocab X Embedding Size
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        gc.collect()

        #With the newly created embedding matrix, we'll fill it up with the words that we have in both 
        #our own dictionary and loaded pretrained embedding. 
        embeddedCount = 0
        for word, i in tokenizer.word_index.items():
            i-=1
            #then we see if this word is in glove's dictionary, if yes, get the corresponding weights
            embedding_vector = embeddings_index.get(word)
            #and store inside the embedding matrix that we will train later on.
            if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector
                embeddedCount+=1
        print('total embedded:',embeddedCount,'common words')
        
        del(embeddings_index)
        gc.collect()
        
        #finally, return the embedding matrix
        return embedding_matrix


# + {"_cell_guid": "df4161dc-6900-4be0-ab99-0dd0c79afa1c", "_uuid": "b4981dc087c111871b303867491fc7480316c6f2", "cell_type": "markdown"}
# The function would return a new embedding matrix that has the loaded weights from the pretrained embeddings for the common words we have, and randomly initialized numbers that has the same mean and standard deviation for the rest of the weights in this matrix.

# + {"_cell_guid": "77c36d68-86d4-4b7f-8099-c8acad87d116", "_uuid": "5482edd9abbcaf24e136a2c3b50f5d350cc01526", "cell_type": "markdown"}
# Let's move on and load our first embeddings from Word2Vec.

# + {"_cell_guid": "a117696b-2428-4692-baff-cc2adc2e2793", "_uuid": "8709b486e16f2f8d622afdcc3715025bd69e785e"}

embedding_matrix = loadEmbeddingMatrix('word2vec')

# + {"_cell_guid": "a4e7b312-a91e-49e1-8131-46655ef8b9b0", "_uuid": "33f70961197d4e7f2e2c4ac77e5c48476b98f166"}
embedding_matrix.shape

# + {"_cell_guid": "287bc31e-bc0e-4fcb-9aa7-f85f20f2c97d", "_uuid": "842265bfbc07fc3242b5f437bce2bd5455bbfe67", "cell_type": "markdown"}
# With the embedding weights, we can proceed to build a LSTM layer. The whole architecture is pretty much the same as the previous one I have done in the earlier kernel here, except that I have turned the LSTM into a bidirectional one, and added a dropout factor to it. 
#
# We start off with defining our input layer. By indicating an empty space after comma, we are telling Keras to infer the number automatically.

# + {"_cell_guid": "31ccdd79-0328-415c-9847-281797c9d8b9", "_uuid": "9b749dbead1781b44b0633660971e27422b3965d"}
inp = Input(shape=(maxlen, )) #maxlen=200 as defined earlier

# + {"_cell_guid": "0b70983e-8ff3-4bcc-ab05-638ad378efdb", "_uuid": "f2d9d3b8e8d8a2a89b4d9de6c497c9a287676eb5", "cell_type": "markdown"}
# Next, we pass it to our Embedding layer, where we use the "weights" parameter to indicate the use of the pretrained embedding weights we have loaded and the "trainable" parameter to tell Keras **not to retrain** the embedding layer.

# + {"_cell_guid": "67b1050c-22e4-492d-8275-918ea41eee57", "_uuid": "eb7bd69eead6b6e1ad872ac55e717bc052f2c41b"}
x = Embedding(len(tokenizer.word_index), embedding_matrix.shape[1],weights=[embedding_matrix],trainable=False)(inp)

# + {"_cell_guid": "ff389666-17e9-4fc4-b3c7-cb663486b69d", "_uuid": "de48b118597ee1998720e43ee04b5bae26d9ece8", "cell_type": "markdown"}
# Next, we pass it to a LSTM unit. But this time round, we will be using a Bidirectional LSTM instead because there are several kernels which shows a decent gain in accuracy by using Bidirectional LSTM.
#
# How does Bidirectional LSTM work? 
#
# ![](https://i.imgur.com/jaKiP0S.png)
#
# Imagine that the LSTM is split between 2 hidden states for each time step. As the sequence of words is being feed into the LSTM in a forward fashion, there's another reverse sequence that is feeding to the different hidden state at the same time. You might noticed later at the model summary that the output dimension of LSTM layer has doubled to 120 because 60 dimensions are used for forward, and another 60 are used for reverse.
#
# The greatest advantage in using Bidirectional LSTM is that when it runs backwards you preserve information from the future and using the two hidden states combined, you are able in any point in time to preserve information from both past and future.
#

# + {"_cell_guid": "1b603886-8c45-4c08-8981-444ac49db52d", "_uuid": "f1c57053cff2aefc1ccdb91a4fa368fff6a817b9", "cell_type": "markdown"}
# We are also introducing 2 more new mechanisms in this notebook: **LSTM Drop out and recurrent drop out.**
#
# Why are we using dropout? You might have noticed that it's easy for LSTM to overfit, and in my previous notebook, overfitting problem starts to surface in just 2 epochs! Drop out is not something new to most of us, and these mechanisms applies the same dropout principles in a LSTM context.
#
# ![](https://i.imgur.com/ksSyArD.png)
# LSTM Dropout is a probabilistic drop out layer on the inputs in each time step, as depict on the left diagram(arrows pointing upwards). On the other hand, recurrent drop out is something like a dropout mask that applies drop out between the hidden states throughout the recursion of the whole LSTM network, which is depicted on the right diagram(arrows pointing to the right). 
#
# These mechanisms could be set via the "dropout" and "recurrent_dropout" parameters respectively. Please ignore the colors in the picture.
#

# + {"_cell_guid": "2fe1b826-8563-486e-9021-0fcd3c3359e5", "_uuid": "149a9300001c76688d2ce7e937f6432d1568ce54"}
x = Bidirectional(LSTM(60, return_sequences=True,name='lstm_layer',dropout=0.1,recurrent_dropout=0.1))(x)

# + {"_cell_guid": "fee7ddd7-42ed-4bf1-8125-032ff435c62e", "_uuid": "faa5c99f9d535cf6e2eb75711d380581a109a904", "cell_type": "markdown"}
# Okay! With the LSTM behind us, we'll feed the output into the rest of the layers which we have done so in the previous kernel. 

# + {"_cell_guid": "c9d4b23a-11af-4f75-b524-f11939a5a014", "_uuid": "8e0c3a6af658f1949281afab2265076de1b0364a"}
x = GlobalMaxPool1D()(x)

# + {"_cell_guid": "6fa4dfe5-3427-4366-a230-2f8d928cabb2", "_uuid": "b0b9f38991a43d5e5c7f55522885503399895122"}
x = Dropout(0.1)(x)

# + {"_cell_guid": "5d2c4062-0e56-4dd3-8980-27a73be2957e", "_uuid": "d3c625875a39e287560c0fae5ce09e0393a961d4"}
x = Dense(50, activation="relu")(x)

# + {"_cell_guid": "e909dc64-e503-4994-9051-a683c316aa3b", "_uuid": "d5c7a2277cc5e13124df0643006f11a1af4b77af"}
x = Dropout(0.1)(x)

# + {"_cell_guid": "cd227bc9-f427-4b0b-84a4-045e3896e572", "_uuid": "b4305789e22f3d84bc86c4388d71a02291b2a7b0"}
x = Dense(6, activation="sigmoid")(x)

# + {"_cell_guid": "473f2792-fe5b-4c4c-af50-db6791b39469", "_uuid": "1f4676b90593995be22ac6bdbb97d45e9c9d366c"}
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

# + {"_cell_guid": "e9c0aacd-3445-4026-a765-612e9e4ebdcb", "_uuid": "b71a23832f2512afa85197f9e7b27519f670de15", "cell_type": "markdown"}
# It's a good idea to see the whole architecture of the network before training as you wouldn't want to waste your precious time training on the wrong set-up.

# + {"_cell_guid": "2a08a9bb-1b4b-4ff5-9dfe-1283ccada5a4", "_uuid": "cd8527ac7eedc31f418273e1eb983aec560b88ed"}
model.summary()

# + {"_cell_guid": "750147b1-f662-4718-9e01-d9d63a8b8514", "_uuid": "12f75dc67eeb70fe7280b24e246a643a98421cb6", "cell_type": "markdown"}
# Finally, we fire off the training process by aiming to run for 4 epochs with a batch size of 32. We save the training and validation loss in a variable so we can take a look and see if there's overfitting.

# + {"_cell_guid": "04479467-b52d-4d7f-a9f6-b7d8c8c33ca3", "_uuid": "e494a8e8fce37b50fbe85442927bc31e35bed98d"}
#batch_size = 32
#epochs = 4
#hist = model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# + {"_cell_guid": "d19f536e-d207-474b-9920-2e132142df57", "_uuid": "5714bec76c510752db51bd7d5bacd22afb473033", "cell_type": "markdown"}
# The training of the model will take longer than what Kaggle kenel allows. I have pre-run it, and this is the result that you should roughly see

# + {"_cell_guid": "43b7c8fa-ad6d-4124-9ab1-ac04e6279497", "_uuid": "d2019be5c4d84c31188ca47887f452d3d0cd11c5", "cell_type": "markdown"}
# Train on 143613 samples, validate on 15958 samples
#
# Epoch 1/4
# 143613/143613 [==============================] - 2938s 20ms/step - loss: 0.0843 - acc: 0.9739 - val_loss: 0.0630 - val_acc: 0.9786
#
# Epoch 2/4
# 143613/143613 [==============================] - 3332s 23ms/step - loss: 0.0573 - acc: 0.9805 - val_loss: 0.0573 - val_acc: 0.9803
#
# Epoch 3/4
# 143613/143613 [==============================] - 3119s 22ms/step - loss: 0.0513 - acc: 0.9819 - val_loss: 0.0511 - val_acc: 0.9817
#
# Epoch 4/4
# 143613/143613 [==============================] - 3137s 22ms/step - loss: 0.0477 - acc: 0.9827 - val_loss: 0.0498 - val_acc: 0.9820
#

# + {"_cell_guid": "7e7690df-5e78-4cca-a302-99c141872755", "_uuid": "26ef45fe790de66cd365c8f468929afdad8fa3b9", "cell_type": "markdown"}
# The result isn't too shabby but it's about the same as the baseline model which we train our own embedding. What about the other pretrained embeddings such as GLOVE and FastText? Let's try them out.
#
# Over here, we are not going to repeat the whole process again. If you are running the notebook yourself, simply replace

# + {"_cell_guid": "b6859878-1f9f-489d-b9c3-0ac6ceded5bf", "_uuid": "163f0a5c118c33ba51b9aaf1f5894cd2e0544a68"}
#loadEmbeddingMatrix('word2vec')

# + {"_cell_guid": "bc93b72f-6ac2-4c7b-886f-c2284ce76d60", "_uuid": "6097649bdc663a9f5523a0ea1a731e82a7f75a93", "cell_type": "markdown"}
# with

# + {"_cell_guid": "1ffc829a-4cfa-4908-ad47-3d350cb77f8b", "_uuid": "f05e7d3ad4a1db1f4e45f034fe10c5df64598d1b"}
#loadEmbeddingMatrix('glove') #for GLOVE or
#loadEmbeddingMatrix('fasttext') #for fasttext

# + {"_cell_guid": "7d506a9a-89b8-4e34-9b40-f5e42c86485e", "_uuid": "1ce3e73d7660de0a04371586fbca4a0e2be62abe", "cell_type": "markdown"}
# to load the pretrained embedding from the different sources. For the sake of our benchmarking. I have pre-run it and collected all the results.

# + {"_cell_guid": "8870888a-3c1a-4d95-98c3-b2f02b725aeb", "_uuid": "f2f10a0c3ee5e1d0fa3aada151b0133a555fb921", "cell_type": "markdown"}
# **GLOVE:**
#
# Train on 143613 samples, validate on 15958 samples
#
# Epoch 1/4
# 143613/143613 [==============================] - 2470s 17ms/step - loss: 0.1160 - acc: 0.9656 - val_loss: 0.0935 - val_acc: 0.9703
#
# Epoch 2/4
# 143613/143613 [==============================] - 2448s 17ms/step - loss: 0.0887 - acc: 0.9721 - val_loss: 0.0800 - val_acc: 0.9737
#
# Epoch 3/4
# 143613/143613 [==============================] - 2410s 17ms/step - loss: 0.0799 - acc: 0.9745 - val_loss: 0.0753 - val_acc: 0.9757
#
# Epoch 4/4
# 143613/143613 [==============================] - 2398s 17ms/step - loss: 0.0753 - acc: 0.9760 - val_loss: 0.0724 - val_acc: 0.9768
#

# + {"_cell_guid": "547b15b9-ce58-406a-b112-5e685663a5ed", "_uuid": "580521b1ce8f9e2d191371623e54e641bc9d641b", "cell_type": "markdown"}
# **Fasttext:**
#
# Train on 143613 samples, validate on 15958 samples
#
# Epoch 1/4
# 143613/143613 [==============================] - 2800s 19ms/step - loss: 0.0797 - acc: 0.9757 - val_loss: 0.0589 - val_acc: 0.9795
#
# Epoch 2/4
# 143613/143613 [==============================] - 2756s 19ms/step - loss: 0.0561 - acc: 0.9808 - val_loss: 0.0549 - val_acc: 0.9804
#
# Epoch 3/4
# 143613/143613 [==============================] - 2772s 19ms/step - loss: 0.0507 - acc: 0.9819 - val_loss: 0.0548 - val_acc: 0.9811
#
# Epoch 4/4
# 143613/143613 [==============================] - 2819s 20ms/step - loss: 0.0474 - acc: 0.9828 - val_loss: 0.0507 - val_acc: 0.9817
#

# + {"_cell_guid": "27fb8058-9a8c-4701-8e80-8739362da2aa", "_uuid": "7e18fa0d3ce07212991b44d28d936bc6c3abb288", "cell_type": "markdown"}
# And of course, the same **baseline** model which doesn't use any pretrained embeddings, taken straight from the previous kenel except that we ran for 4 epochs:

# + {"_cell_guid": "4c5668f7-e667-4c66-9e33-733308813d16", "_uuid": "ef52c54c1be94b232e1cf75d0ce584b67797af3d", "cell_type": "markdown"}
# Train on 143613 samples, validate on 15958 samples
#
# Epoch 1/4
# 143613/143613 [==============================] - 5597s 39ms/step - loss: 0.0633 - acc: 0.9788 - val_loss: 0.0480 - val_acc: 0.9825
#
# Epoch 2/4
# 143613/143613 [==============================] - 5360s 37ms/step - loss: 0.0448 - acc: 0.9832 - val_loss: 0.0464 - val_acc: 0.9828
#
# Epoch 3/4
# 143613/143613 [==============================] - 5352s 37ms/step - loss: 0.0390 - acc: 0.9848 - val_loss: 0.0470 - val_acc: 0.9829
#
# Epoch 4/4
# 129984/143613 [==============================] - 5050s 37ms/step - loss: 0.0386 - acc: 0.9858 - val_loss: 0.0478 - val_acc: 0.9830

# + {"_cell_guid": "bf5029ef-a106-491e-9d23-b57ad0ef78df", "_uuid": "893c8525aa5c55146e7b1bbf61a50d65d5bf9891", "cell_type": "markdown"}
# It's easier if we plot the losses into graphs.

# + {"_cell_guid": "fcd1091f-39a5-402f-a15b-fbc0c1c0b083", "_uuid": "c1208a4158680624a9899fa34a689824bf1e1e83"}
all_losses = {
'word2vec_loss': [0.084318213647104789,
  0.057314205012433353,
  0.051338302593577821,
  0.047672802178572039],
 'word2vec_val_loss': [0.063002561892695971,
  0.057253835496480658,
  0.051085027624451551,
  0.049801279793734249],
'glove_loss': [0.11598931579683543,
  0.088738223480436862,
  0.079895263566000005,
  0.075343037429358703],
 'glove_val_loss': [0.093467933030432285,
  0.080007083813922117,
  0.075349041991106688,
  0.072366507668134517],
 'fasttext_loss': [0.079714499498945865,
  0.056074704045674786,
  0.050703874653286324,
  0.047420131195761134],
 'fasttext_val_loss': [0.058888281775148932,
  0.054906051694414926,
  0.054768857866843601,
  0.050697043558286421],
 'baseline_loss': [0.063304489498915865,
  0.044864004045674786,
  0.039013874651286124,
  0.038630130175761134],
 'baseline_val_loss': [0.048044281075148932,
  0.046414051594414926,
  0.047058757860843601,
  0.047886043558285421]
}

# + {"_cell_guid": "a9a12287-3a36-4337-863f-bea51636f401", "_uuid": "70f5973bd6836507e0ffebaa86a9f40c4c50e409"}
#f, ax = plt.subplots(1)
epochRange = np.arange(1,5,1)
plt.plot(epochRange,all_losses['word2vec_loss'])
plt.plot(epochRange,all_losses['glove_loss'])
plt.plot(epochRange,all_losses['fasttext_loss'])
plt.plot(epochRange,all_losses['baseline_loss'])
plt.title('Training loss for different embeddings')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Word2Vec', 'GLOVE','FastText','Baseline'], loc='upper left')
plt.show()

# + {"_cell_guid": "0b9a0aba-9369-46f2-b133-195202f2e9a6", "_uuid": "e9579289104243d649dc73744de3358107bcdb8b", "cell_type": "markdown"}
# Well, it certainly looks like the baseline has the minimum training loss. But before we close this case and pick the baseline model as the winner, this plot does not tell the full story as there seems to be some overfitting in the baseline model. It appears that from the 2nd epoch, overfitting has started to slip in as the validation loss has become higher than training loss.

# + {"_cell_guid": "fcdf6818-84af-4ddc-930e-434ee7e5fe76", "_uuid": "52bdc81843573706c6c50fed9af650b0a23bced6"}
epochRange = np.arange(1,5,1)
plt.plot(epochRange,all_losses['baseline_loss'])
plt.plot(epochRange,all_losses['baseline_val_loss'])
plt.title('Training Vs Validation loss for baseline model')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

# + {"_cell_guid": "1f867f4f-e6dc-4d2c-bf41-853bb8f3754e", "_uuid": "90e098f1d571304ebd9752eb0318b209ed1a7f9e", "cell_type": "markdown"}
# What about the rest? Let's plot all the training/validation loss plots out to compare side by side.

# + {"_cell_guid": "03aef930-d6d7-493f-9d2f-77894a28b889", "_uuid": "e5ada61266f2f282fcef77415f9c9c069e274a16"}
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row',figsize=(20, 20))

plt.title('Training Vs Validation loss for all embeddings')
ax1.plot(epochRange,all_losses['baseline_loss'])
ax1.plot(epochRange,all_losses['baseline_val_loss'])
ax1.set_title('Baseline')
ax1.set_ylim(0.03, 0.12)

ax2.plot(epochRange,all_losses['word2vec_loss'])
ax2.plot(epochRange,all_losses['word2vec_val_loss'])
ax2.set_title('Word2Vec')
ax2.set_ylim(0.03, 0.12)

ax3.plot(epochRange,all_losses['glove_loss'])
ax3.plot(epochRange,all_losses['glove_val_loss'])
ax3.set_title('GLOVE')
ax3.set_ylim(0.03, 0.12)


ax4.plot(epochRange,all_losses['fasttext_loss'])
ax4.plot(epochRange,all_losses['fasttext_val_loss'])
ax4.set_title('FastText')
ax4.set_ylim(0.03, 0.12)

plt.show()

# + {"_cell_guid": "54869c0c-be90-4c06-966f-1b7430b1905d", "_uuid": "6c2ea50ea42d22e49a214159a7629349ba360603", "cell_type": "markdown"}
# With all the losses laid out, it's easy to see which the best option is. While it appears that GLOVE still some room to go before it overfits, the loss is high compared to the rest. On the other hand, Word2Vec and FastText starts to overfit at the 4rd and 3rd epochs respectively. So which one would you pick as the winner? In my opinion, **still the baseline model.**
#
# So what went wrong? Aren't pretrained embeddings supposed to improve because it's trained with billions of words from tons of feature-rich corpus?
#
# One probability is that these pretrained embeddings are not trained against text in the same context so the number of common words between our text and text that these pretrained embeddings were trained would be low. Let's plot the number of words we are using in the embedding layer.

# + {"_cell_guid": "565c40d6-5ac6-455e-a850-233fad242cc1", "_uuid": "7ef543ff4da779bf7b2f647af668b8a9b5c691b1"}
wordCount = {'word2vec':66078,'glove':81610,'fasttext':59613,'baseline':210337}

# + {"_cell_guid": "2c192a5c-9c3f-40dd-9017-046276595af0", "_uuid": "62052a6836c4fa89e32c9b4834e61f7a14e6599f"}
ind = np.arange(0,4,1)  # the x locations for the groups
width = 0.35       # the width of the bars

plt.title('Number of common words used in different embeddings')
embNames = list(wordCount.keys())
embVals = list(wordCount.values())
plt.barh(ind,embVals,align='center', height=0.5, color='m',tick_label=embNames)
plt.show()

# + {"_cell_guid": "6e47fdc5-48d7-44aa-a3ec-ace4685b0d80", "_uuid": "eddcba16789be0b6b7a88daddd1cd4aef86d456e", "cell_type": "markdown"}
# From the above bar chart, it's obvious that the baseline would have the most words since the embedding layer is trained using the words in the dataset. The important takeaway is that the pretrained embeddings only contains about 60,000 words in common(less than half of baseline) and the embedding layer that is built from these pretrained weights couldn't represent the training data well enough.
#
# Although building your own embedding takes a longer time, it might be worthwhile because it builds specifically for your context.

# + {"_cell_guid": "884e7b47-a91d-4181-8d87-f6ccf205374d", "_uuid": "9d2ac84e88f815744e51f6ade260712ddfbd58a5", "cell_type": "markdown"}
# And that finally wraps up this kernel! I hope someone learnt something in this kernel. If you spot an error, feel free to let me know by commenting below. 
#
# Thanks for reading and good luck in the competition!

# + {"_cell_guid": "9f65702d-6d69-41d1-bd14-ff05aa9e871a", "_uuid": "2c461db8c144e5cef8e4b3f97dda1f2970e15660", "cell_type": "markdown"}
# **TODO:**
# 1. There are many pretrained embeddings in Kaggle, and they are trained in different contexts of text corpus. You could try out other pretrained embeddings that is more suitable to the dataset in our competition.
# 2. Introduce LSTM drop out and recurrent drop out in baseline model, and tune the dropout rate to decrease overfitting.

# + {"_cell_guid": "8d94ab52-7adb-4b6b-b8ed-2db2aaac7321", "_uuid": "caf5fe6d19422f715fb9ebb63f5d0e123cb3650d"}

