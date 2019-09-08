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

# + {"_cell_guid": "1cc94569-8d1f-4bd1-802b-78a7ae487ef1", "_uuid": "2ae1b895eb4f40ddbfd6ec15700aa9b414fbb5bf", "cell_type": "markdown"}
# ![](https://www.pyimagesearch.com/wp-content/uploads/2017/12/not_santa_detector_dl_logos.jpg)
# **This notebook attempts to tackle this classification problem by using Keras LSTM. While there are many notebook out there that are already tackling using this approach, I feel that there isn't enough explanation to what is going on each step. As someone who has been using vanilla Tensorflow, and recently embraced the wonderful world of Keras, I hope to share with fellow beginners the intuition that I gained from my research and study. **
#
# **Join me as we walk through it. **

# + {"_cell_guid": "9c0cc9c2-ba1f-46ad-bec7-47a437e65d16", "_uuid": "c1683e580b8e20f8fb94dc46444695d70efdee57", "cell_type": "markdown"}
# We import the standard Keras library

# + {"_cell_guid": "463f0f87-f96f-435c-8f40-d39dfef8dc36", "_uuid": "09df08cb0050aa8fd4b5b7bd6606b4d79b7a9d08"}
import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

# + {"_cell_guid": "42634224-3ba5-41e3-8612-285cc8a8af16", "_uuid": "5e88facdabb090f8eaf48d014838e08cde63c455", "cell_type": "markdown"}
# Loading the train and test files, as usual

# + {"_cell_guid": "62c89f51-8315-4f50-97c9-de3539e884a9", "_uuid": "447247729764c3579a4a2d6bf69287abff0b9af1"}
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# + {"_cell_guid": "d7ab4de0-419e-4b65-9191-d211bfb80cd5", "_uuid": "a1cbcc835faa99b426d229f6b7c708e84c57278d", "cell_type": "markdown"}
# A sneak peek at the training and testing dataset

# + {"_cell_guid": "2b307015-414b-466e-9a80-72b9f1b61df8", "_uuid": "9a4a51116882f985fb362263a1346bf1883a3f04"}
train.head()

# + {"_cell_guid": "24fffe7d-54cf-42bd-a6a9-c8a15f26adef", "_uuid": "47f9d0bd297310fe44cd57ec3125beb6b5520bc1", "cell_type": "markdown"}
# A common preprocessing step is to check for nulls, and fill the null values with something before proceeding to the next steps. If you leave the null values intact, it will trip you up at the modelling stage later

# + {"_cell_guid": "eda2a4d3-c4d4-4d09-a9f3-d3087b0ef96f", "_uuid": "f5c25b08f4b5504ab984fc8a7b3748f8cef42440"}
train.isnull().any(),test.isnull().any()

# + {"_cell_guid": "311298e8-0dd3-4fac-882b-60ac2fdce607", "_uuid": "40a2fe919d61c221c5302c77b12fbf4bf17cd740", "cell_type": "markdown"}
# Looks like we don't need to deal with the null values after all!
#
# Note that: There are tons of preprocessing and feature engineering steps you could do for the dataset, but our focus today is not about the preprocessing task so what we are doing here is the minimal that could get the rest of the steps work well.

# + {"_cell_guid": "56b16fae-1e7f-49c9-9712-e207a24fbf6f", "_uuid": "a0df206844644222500da46147922b02f6982ec6", "cell_type": "markdown"}
# Movng on, as you can see from the sneak peek, the dependent variables are in the training set itself so we need to split them up, into X and Y sets.

# + {"_cell_guid": "4bf0415e-6dcc-40c1-8d04-96b1ac95eb27", "_uuid": "548389f5096016fcb405fc73d5c21390d6c72d83"}
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_train = train["comment_text"]
list_sentences_test = test["comment_text"]

# + {"_cell_guid": "2f7c10ea-8146-48ce-b99d-35b3c785ebd8", "_uuid": "b5acead06f62cc6328488b155ac79148552ed07b", "cell_type": "markdown"}
# The approach that we are taking is to feed the comments into the LSTM as part of the neural network but we can't just feed the words as it is. 
#
# So this is what we are going to do:
# 1. Tokenization - We need to break down the sentence into unique words. 
#     For eg, "I love cats and love dogs" will become ["I","love","cats","and","dogs"]
# 2. Indexing - We put the words in a dictionary-like structure and give them an index each
#     For eg, {1:"I",2:"love",3:"cats",4:"and",5:"dogs"}
# 3. Index Representation- We could represent the sequence of words in the comments in the form of index, and feed this chain of index into our LSTM.
#     For eg, [1,2,3,4,2,5] 

# + {"_cell_guid": "385d8dc8-1e5f-47dc-b49b-bc370cb4d90f", "_uuid": "b0ac72c6c2b2fef2604bff98985a1f175f680207", "cell_type": "markdown"}
# Fortunately, Keras has made our lives so much easier. If you are using the vanilla Tensorflow, you probably need to implement your own dictionary structure and handle the indexing yourself. In Keras, all the above steps can be done in 4 lines of code. Note that we have to define the number of unique words in our dictionary when tokenizing the sentences.

# + {"_cell_guid": "0d373763-f5c6-4d28-a1d7-f5030d47320f", "_uuid": "8d814a2dadca32a8810dca02c7d10dff38a84660"}
max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

# + {"_cell_guid": "9aa4366b-d20c-4404-afb0-3c2a8575fb12", "_uuid": "cac3f1c1791515820aae71373ea123f5aff3bc65", "cell_type": "markdown"}
# You could even look up the occurrence and the index of each words in the dictionary:

# + {"_cell_guid": "3c09fa4e-9863-4d49-aa26-ec213a80e500", "_uuid": "24c8936770226aafac6f868ace63f68718b3f18e"}
#commented it due to long output
#for occurence of words
#tokenizer.word_counts
#for index of words
#tokenizer.word_index
# -

to = dict(tokenizer.word_counts)
sorted(to.items(), key= lambda x: x[1], reverse=True)

# + {"_cell_guid": "29932a7d-9b81-471c-93df-317a19a7414b", "_uuid": "a594384bb2ad587d62264aa238724067a022bad4", "cell_type": "markdown"}
# Now, if you look at "list_tokenized_train", you will see that Keras has turned our words into index representation for us

# + {"_cell_guid": "c24422b6-f4a5-4c90-9e1f-3492d5790036", "_uuid": "252b3a9ac2fd63901b7d6cbd78033230e000198c"}
list_tokenized_train[:1]

# + {"_cell_guid": "1057990a-4e2b-4580-8782-5bfa30f01322", "_uuid": "275647d234cfd572d1200384feee1c292d16925c", "cell_type": "markdown"}
# But there's still 1 problem! What if some comments are terribly long, while some are just 1 word? Wouldn't our indexed-sentence look like this:
#    
#    Comment #1: [8,9,3,7,3,6,3,6,3,6,2,3,4,9]
#   
#   Comment #2: [1,2]
#
# And we have to feed a stream of data that has a consistent length(fixed number of features) isn't it?

# + {"_cell_guid": "6cc19fbd-3034-42e3-a837-cedfde6a839f", "_uuid": "8afd0371bef4f9954c586547f2400b71ced0bfb2", "cell_type": "markdown"}
# And this is why we use "padding"! We could make the shorter sentences as long as the others by filling the shortfall by zeros.But on the other hand, we also have to trim the longer ones to the same length(maxlen) as the short ones. In this case, we have set the max length to be 200.

# + {"_cell_guid": "d5eca775-e326-421c-a90b-2fc69f14020e", "_uuid": "77cb38be6af621207460fa657df85bc70535a21c"}
maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

# + {"_cell_guid": "e6856558-10b2-44b9-a347-5e93add93cff", "_uuid": "b5df6c73849f02ab4f58698d73b7d74a44829f0a", "cell_type": "markdown"}
# How do you know what is the best "maxlen" to set? If you put it too short, you might lose some useful feature that could cost you some accuracy points down the path.If you put it too long, your LSTM cell will have to be larger to store the possible values or states.
#
# One of the ways to go about it is to see the distribution of the number of words in sentences.

# + {"_cell_guid": "3c447319-2453-48ed-8e68-0c192c0873f6", "_uuid": "69fde7d5c0ff270393321d28298610e2d3a56634"}
totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]

# + {"_cell_guid": "e522d821-136f-4b07-80f4-d1d9323f5ef4", "_uuid": "4116cf21a3a94d2b05017c9ae8f3b773317a71c0"}
plt.hist(totalNumWords,bins = np.arange(0,410,10))#[0,50,100,150,200,250,300,350,400])#,450,500,550,600,650,700,750,800,850,900])
plt.show()

# + {"_cell_guid": "ab1dec2f-65b8-456f-9f29-4a85db23eb64", "_uuid": "05dbfe5778b3f5554f437f8487833bba3147bd0d", "cell_type": "markdown"}
# As we can see, most of the sentence length is about 30+. We could set the "maxlen" to about 50, but I'm being paranoid so I have set to 200. Then again, it sounds like something you could experiment and see what is the magic number.

# + {"_cell_guid": "b360bb42-4779-410a-a171-e5bd9ef148ce", "_uuid": "0b3389d55e4ba925a2a873acdd2277cc2f2818b4", "cell_type": "markdown"}
# **Finally the start of building our model!**

# + {"_cell_guid": "ca69f395-cb69-433d-a486-c1795ab8d722", "_uuid": "dd459005174e11a82666db7ec198090a528533ff", "cell_type": "markdown"}
# This is the architecture of the model we are trying to build. It's always to good idea to list out the dimensions of each layer in the model to think visually and help you to debug later on.
# ![](https://i.imgur.com/txJomEa.png)

# + {"_cell_guid": "29f48f2e-7d01-4532-886c-1e5110afa262", "_uuid": "fb44734c3705f6a920a25ba2d5f13aa30524859e", "cell_type": "markdown"}
# As mentioned earlier, the inputs into our networks are our list of encoded sentences. We begin our defining an Input layer that accepts a list of sentences that has a dimension of 200.
# ![](https://i.imgur.com/uSjU4J7.png)
# By indicating an empty space after comma, we are telling Keras to infer the number automatically.

# + {"_cell_guid": "f833649b-7a62-4480-9ee9-193128eefb9b", "_uuid": "a8578e2b581989b196b3f8296995c3241f32f1d5"}
inp = Input(shape=(maxlen, )) #maxlen=200 as defined earlier

# + {"_cell_guid": "12bf4a50-3e92-4b32-bc77-31dc422a36b2", "_uuid": "f390a234ee8f61f1532bf06ada0da27e225615f1", "cell_type": "markdown"}
# Next, we pass it to our Embedding layer, where we project the words to a defined vector space depending on the distance of the surrounding words in a sentence. Embedding allows us to reduce model size and most importantly the huge dimensions we have to deal with, in the case of using one-hot encoding to represent the words in our sentence.
# ![](https://www.tensorflow.org/versions/r0.12/images/embedding-custom-projection.png)
# The output of the Embedding layer is just a list of the coordinates of the words in this vector space. For eg. (-81.012) for "cat" and  (-80.012) for "dog". We could also use the distance of these coordinates to detect relevance and context. Embedding is a pretty deep topic, and if you are interested, this is a comprehensive guide: https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/

# + {"_cell_guid": "b08562cd-9726-4692-bddf-f9dcdf805633", "_uuid": "f1914debafd41f7f1a9f3935f2e8371796db84e4", "cell_type": "markdown"}
# We need to define the size of the "vector space" we have mentioned above, and the number of unique words(max_features) we are using. Again, the embedding size is a parameter that you can tune and experiment.

# + {"_cell_guid": "23281f14-130c-41d3-98cc-49283ff27177", "_uuid": "9bfea70ef8209f178c9974c61bef16e7ab1bf020"}
embed_size = 128
x = Embedding(max_features, embed_size)(inp)

# + {"_kg_hide-output": true, "_cell_guid": "9afef8d3-c424-4d71-8e26-761a733711c4", "_kg_hide-input": true, "_uuid": "2859d2e8687d37ab5a2d2852a06a532b905a7cd2", "cell_type": "markdown"}
# The embedding layer outputs a 3-D tensor of (None, 200, 128). Which is an array of sentence(None means that it's size is inferred), and for each words(200), there is an array of 128 coordinates in the vector space of embedding. 

# + {"_cell_guid": "e442fd20-f341-4e18-bad4-dd8d7467185c", "_uuid": "89def7983c45e7afd7c874bdb404ee3d59621ce3", "cell_type": "markdown"}
# Next, we feed this Tensor into the LSTM layer. We set the LSTM to produce an output that has a dimension of 60 and want it to return the whole unrolled sequence of results.
# As you probably know, LSTM or RNN works by recursively feeding the output of a previous network into the input of the current network, and you would take the final output after X number of recursion. But depending on use cases, you might want to take the unrolled, or the outputs of each recursion as the result to pass to the next layer. And this is the case.

# + {"_cell_guid": "95339082-00f5-4e59-a0da-c91118bd6fe3", "_uuid": "487913bed9104e5610b0f47884997b2bb4b4b9ea", "cell_type": "markdown"}
# ![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)

# + {"_cell_guid": "9237b661-c661-4408-945f-e7c468b287db", "_uuid": "4559a488e7c4efbdcaab54c618d5d1c236281fd4", "cell_type": "markdown"}
# From the above picture, the unrolled LSTM would give us a set of h0,h1,h2 until the last h.

# + {"_cell_guid": "90b45870-39c0-46c3-bbdb-402af7ea3434", "_uuid": "1ff0311517159bc5a0af30ca7c526425b86d3ad6", "cell_type": "markdown"}
# From the short line of code that defines the LSTM layer, it's easy to miss the required input dimensions. LSTM takes in a tensor of [Batch Size, Time Steps, Number of Inputs]. Batch size is the number of samples in a batch, time steps is the number of recursion it runs for each input, or it could be pictured as the number of "A"s in the above picture. Lastly, number of inputs is the number of variables(number of words in each sentence in our case) you pass into LSTM as pictured in "x" above.

# + {"_cell_guid": "eb970d85-d3f4-4076-92b2-b7a4e5f3270b", "_uuid": "2cbce9ccceefb0d9ec2328976a6685c21e5ad181", "cell_type": "markdown"}
# We can make use of the output from the previous embedding layer which outputs a 3-D tensor of (None, 200, 128) into the LSTM layer. What it does is going through the samples, recursively run the LSTM model for 200 times, passing in the coordinates of the words each time. And because we want the unrolled version, we will receive a Tensor shape of (None, 200, 60), where 60 is the output dimension we have defined.

# + {"_cell_guid": "47c448cf-773d-47c8-a006-b761b0fabbf1", "_uuid": "9e4dd0350f3d779a7c05f04dbc3406e779b72f17"}
x = LSTM(60, return_sequences=True,name='lstm_layer')(x)

# + {"_cell_guid": "5dfe53cc-5d27-42bb-bcbb-55738028da51", "_uuid": "fdfe0d52ae736726643c21e6a8068ab50b0f53ba", "cell_type": "markdown"}
# Before we could pass the output to a normal layer, we need to reshape the 3D tensor into a 2D one. We reshape carefully to avoid throwing away data that is important to us, and ideally we want the resulting data to be a good representative of the original data.
#
# Therefore, we use a Global Max Pooling layer which is traditionally used in CNN problems to reduce the dimensionality of image data. In simple terms, we go through each patch of data, and we take the maximum values of each patch. These collection of maximum values will be a new set of down-sized data we can use.
#
# As you can see from other Kaggle kernels, different variants (Average,Max,etc) of pooling layers are used for dimensionality reduction and they could yield different results so do try them out. 
#
# If you are interested in finding out the technical details of pooling, read up here: https://wiseodd.github.io/techblog/2016/07/18/convnet-maxpool-layer/
#

# + {"_cell_guid": "9a0fed7f-cbc5-4591-a6b2-30a75090aed2", "_uuid": "e5539d8b185f15a420e9c3f1c800960ce07c1002"}
x = GlobalMaxPool1D()(x)

# + {"_cell_guid": "ae193bb2-5e0f-40e1-bcbb-4be3993687ed", "_uuid": "e9e7cc514b0e674d205a07d7931a3486b4c2f7e5", "cell_type": "markdown"}
# With a 2D Tensor in our hands, we pass it to a Dropout layer which indiscriminately "disable" some nodes so that the nodes in the next layer is forced to handle the representation of the missing data and the whole network could result in better generalization.
# ![](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/image_folder_5/dropout.jpeg)
#
# We set the dropout layer to drop out 10%(0.1) of the nodes.

# + {"_cell_guid": "d6bb28c9-3af8-4808-8316-eb3386ca0b93", "_uuid": "85493c90e9358fc3164b9c01904adc6caf531d7d"}
x = Dropout(0.1)(x)

# + {"_cell_guid": "44cc2f04-9807-4535-97e3-d24d932be4f6", "_uuid": "ab410f9a1e5df58195f89d6debd1d95ffb542c20", "cell_type": "markdown"}
# After a drop out layer, we connect the output of drop out layer to a densely connected layer and the output passes through a RELU function. In short, this is what it does:
#
# **Activation( (Input X Weights) + Bias)**
#
# all in 1 line, with the weights, bias and activation layer all set up for you! 
# We have defined the Dense layer to produce a output dimension of 50.

# + {"_cell_guid": "60c40b04-ccaa-4fa2-836f-18260e1379f9", "_uuid": "84773a29b736311fd19fa96a398f65867aa20118"}
x = Dense(50, activation="relu")(x)

# + {"_cell_guid": "a4a70745-b597-4e26-b488-073d5392f92f", "_uuid": "869d7709290e0f29f5151423b56c634b1bc70fa4", "cell_type": "markdown"}
# We feed the output into a Dropout layer again.

# + {"_cell_guid": "8eb0af99-79ad-4900-a74d-b9976b3523d5", "_uuid": "ff7942a49a599810574a75201b7e5c8a046c8733"}
x = Dropout(0.1)(x)

# + {"_cell_guid": "268032f9-3ccd-484c-a964-5a8ae5ee01be", "_uuid": "292f637a48338ddebbf0dc880202af0d30ebc17a", "cell_type": "markdown"}
# Finally, we feed the output into a Sigmoid layer. The reason why sigmoid is used is because we are trying to achieve a binary classification(1,0) for each of the 6 labels, and the sigmoid function will squash the output between the bounds of 0 and 1.

# + {"_cell_guid": "22243974-cce1-4556-b2a2-8cfe97bce70f", "_uuid": "9cea3bbfb39ca0d01bf7c5ebfd0e48f4f998a8e7"}
x = Dense(6, activation="sigmoid")(x)

# + {"_cell_guid": "8c52162f-ab95-436d-a72a-ef82ae4e89a7", "_uuid": "e9df6684a6b32025c0732a572955b5b6b2b8315a", "cell_type": "markdown"}
# We are almost done! All is left is to define the inputs, outputs and configure the learning process. We have set our model to optimize our loss function using Adam optimizer, define the loss function to be "binary_crossentropy" since we are tackling a binary classification. In case you are looking for the learning rate, the default is set at 0.001.

# + {"_cell_guid": "8e1fc640-385d-40be-b32c-5703c2635a78", "_uuid": "d152b94cc890b512ba6afd9b2fdb0a14c8ea76b4"}
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

# + {"_cell_guid": "c2586e4e-33a3-4dcb-a757-1dcfd89aea8c", "_uuid": "4a902e06a59d2ffeecf846fd5a8a2a56471512ad", "cell_type": "markdown"}
# The moment that we have been waiting for as arrived! It's finally time to put our model to the test. We'll feed in a list of 32 padded, indexed sentence for each batch and split 10% of the data as a validation set. This validation set will be used to assess whether the model has overfitted, for each batch. The model will also run for 2 epochs. These are some of the tunable parameters that you can experiment with, to see if you can push the accurate to the next level without crashing your machine(hence the batch size).

# + {"_cell_guid": "8ae6cadb-0543-4fee-9dba-1eb99987c441", "_uuid": "1c26824601d297f618bea32e1e08c3af799fc5a4"}
batch_size = 32
epochs = 2
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# + {"_cell_guid": "402593c0-058c-4436-b3fb-968e992d552a", "_uuid": "5ad5c5b5f17094d714964abc1ff3747f634a942e", "cell_type": "markdown"}
# Seems that the accuracy is pretty decent for a basic attempt! There's a lot that you could do (see TODO below) to further improve the accuracy so feel free to fork the kernel and experiment for yourself!

# + {"_cell_guid": "eeb2aa02-27d3-4045-8633-9a8088446430", "_uuid": "6de2b32e51346614f91df096c35f2d9c0ea0dd3d", "cell_type": "markdown"}
# **Additional tips and tricks**
#
# 1) If you have hit some roadblocks, especially when it starts returning dimension related errors, a good idea is to run "model.summary()" because it lists out all your layer outputs, which is pretty useful for diagnosis.

# + {"_cell_guid": "0192471f-516d-4f3d-9ce0-784da0ca59ab", "_uuid": "5da468e7b497b736510f0521863bc696809e134b"}
model.summary()

# + {"_cell_guid": "cdfd7316-f6db-4095-b4f8-da3b0bb1e796", "_uuid": "87afd9a57213185d5ec33073ef7b40269e73e2e5", "cell_type": "markdown"}
# 2) While adding more layers, and doing more fancy transformations, it's a good idea to check if the outputs are performing as you have expected. You can reveal the output of a particular layer by :

# + {"_cell_guid": "5bdea10d-3c9c-410d-a639-6ec810d36b79", "_uuid": "875aea01dc1439076c37f56095851ed8bd5bc078"}
from tensorflow.keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[2].output])
layer_output = get_3rd_layer_output([X_t[:1]])[0]
layer_output.shape
#print layer_output to see the actual data

# + {"_cell_guid": "67706b68-0c5f-4135-96d9-6ab04f7c6a37", "_uuid": "bd874e5dab3de6117b766039fdd1e28969d0408d", "cell_type": "markdown"}
# Personally I find Keras cuts down a lot of time and saves you the agony of dealing with grunt work of defining the right dimensions for matrices. The time saved could have spent on fruitful tasks like experimenting with different variations of model, etc. However, I find that many variables and processes have been initialized automatically in a way that beginners to deep learning might not realize what is going on under the hood. There's a lot of intricate details so I encourage newbies to open up this black box and you will be rewarded with a wealth of knowledge in deep learning.
#
# I hope someone will find this short guide useful. If you like to see more of such guides, support me by upvoting this kernel. Thanks for reading and best of luck for the competition!
#

# + {"_cell_guid": "cd4eb05b-b8bb-4e0e-8def-d793afc4e907", "_uuid": "0dc5a9c4b3c6459a828bd7be2a96ad08368e4476", "cell_type": "markdown"}
# **TODO:**
# 1. Using Pre-trained models to boost accuracy and take advantage of existing efforts
# 2. Hyper-parameter tuning of bells and whistles
# 3. Introduce early stopping during training of model
# 4. Experiment with different architecture.
#

# + {"_cell_guid": "9f197004-0109-4955-895d-6be3b074d923", "_uuid": "162c28bf0227dd13f8ebb2718981f21d4077ce0d"}

