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

# + {"_cell_guid": "c3e71160-1057-49d1-a210-efbcf57005e9", "_uuid": "838968925befec8e7ae0557312425a9dd6add610", "cell_type": "markdown"}
# # Update: 
# The kernal has been updated for the new test and train datasets.

# + {"_cell_guid": "496d8c4f-566a-4599-abd9-8dfbd477af8a", "_uuid": "b7676bf7ddfb463d335a507359c33c383704cd70", "cell_type": "markdown"}
# # Introduction:
# Being anonymous over the internet can sometimes make people say nasty things that they normally would not in real life.
# Let's filter out the hate from our platforms one comment at a time. 
#
# ## Objective:
# To create an EDA/ feature-engineering starter notebook for toxic comment classification.
#
# ## Data Overview:
# The dataset here is from wiki corpus dataset which was rated by human raters for toxicity.
# The corpus contains 63M comments from discussions relating to user pages and articles dating from 2004-2015. 
#
# Different platforms/sites can have different standards for their toxic screening process. Hence the comments are tagged in the following five categories
# * toxic
# * severe_toxic
# * obscene
# * threat
# * insult
# * identity_hate
#
# The tagging was done via **crowdsourcing** which means that the dataset was rated by different people and the tagging might not be 100% accurate too. The same concern is being discussed [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/46131).
#
# The [source paper](https://arxiv.org/pdf/1610.08914.pdf) also contains more interesting details about the dataset creation.
#
# ## Note:
# A [New test dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/46177) is being created by the organizers as the test set labels are present [here](https://figshare.com/articles/Wikipedia_Talk_Labels_Toxicity/4563973).
#
# The kernal has been updated for the new data.

# + {"_cell_guid": "b3f5968a-4194-4bc6-a0a3-c9e8252e4ce9", "_uuid": "5ee2347c13ad3f58f4dbb409206d73e13bf4bc66"}
#Check the dataset sizes(in MB)
!du -l ../input/*

# + {"_cell_guid": "8380fea7-4a2d-4143-99cf-9aa869aee6a2", "_uuid": "82d1ae7d4666ce7715f504165177ef053b95b719"}
#import required packages
#basics
import pandas as pd 
import numpy as np

#misc
import gc
import time
import warnings

#stats
from scipy.misc import imread
from scipy import sparse
import scipy.stats as ss

#viz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
from wordcloud import WordCloud ,STOPWORDS
from PIL import Image
import matplotlib_venn as venn

#nlp
import string
import re    #for regex
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   


#FeatureEngineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

#settings
start_time=time.time()
color = sns.color_palette()
sns.set_style("dark")
eng_stopwords = set(stopwords.words("english"))
warnings.filterwarnings("ignore")

lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()

# %matplotlib inline

# + {"_cell_guid": "c0708360-0740-46a2-893e-6845df578676", "_kg_hide-input": true, "_uuid": "aaa44a44b460a782ffd10d176251e67004ee43db"}
#importing the dataset
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

# + {"_cell_guid": "b7c6deae-7803-436f-9ebf-beb77bc7b0f3", "_uuid": "94e6f34adeedfe847df77275781bbed95f6a055f"}
#take a peak
train.tail(10)
# -

train['comment_text'][159560]

# + {"_cell_guid": "bc0cb30d-76dd-4676-930c-81f6f33dca63", "_kg_hide-input": true, "_uuid": "e2f0dfd228aa2131ab7c75664fef3c420327060e"}
nrow_train=train.shape[0]
nrow_test=test.shape[0]
sum=nrow_train+nrow_test
print("       : train : test")
print("rows   :",nrow_train,":",nrow_test)
print("perc   :",round(nrow_train*100/sum),"   :",round(nrow_test*100/sum))

# + {"_cell_guid": "87c19794-eec8-4101-8817-47434c41f3c8", "_uuid": "2886eb3f30f72868be2f16c6c648c6ae8ec59c3d", "cell_type": "markdown"}
# There is a 30:70 train: test split and the test set might change in the future too.
#
# Let's take a look at the class imbalance in the train set.
#
# ### Class Imbalance:

# + {"_cell_guid": "1649bd00-1734-4706-ba52-21248a53aa12", "_kg_hide-input": true, "_uuid": "43801836d1809c99ff251999f98bdc146c5344f2"}
x=train.iloc[:,2:].sum()
#marking comments without any tags as "clean"
rowsums=train.iloc[:,2:].sum(axis=1)
train['clean']=(rowsums==0)
#count number of clean entries
train['clean'].sum()
print("Total comments = ",len(train))
print("Total clean comments = ",train['clean'].sum())
print("Total tags =",x.sum())

# + {"_cell_guid": "355f7787-448a-4a8d-b57d-cfe94be6a498", "_uuid": "ce37927f4bdd3926c2b142d05a1a0a263309a8da"}
print("Check for missing values in Train dataset")
null_check=train.isnull().sum()
print(null_check)
print("Check for missing values in Test dataset")
null_check=test.isnull().sum()
print(null_check)
print("filling NA with \"unknown\"")
train["comment_text"].fillna("unknown", inplace=True)
test["comment_text"].fillna("unknown", inplace=True)

# + {"_cell_guid": "2b58d925-9f18-4c28-8c1a-8e37d17ae8b4", "_kg_hide-input": true, "_uuid": "0a3ee7d1bb8c764dd7115657b028c3108b86d08b"}
x=train.iloc[:,2:].sum()
#plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("# per class")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('Type ', fontsize=12)
#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()

# + {"_cell_guid": "dec83630-3c6a-4d51-9862-58093fc61aa0", "_uuid": "cd1e6fa8ba8b24f8a72f678506c8ff9833254cc9", "cell_type": "markdown"}
# * The toxicity is not evenly spread out across classes. Hence we might face class imbalance problems
# * There are ~95k comments in the training dataset and there are ~21 k tags and ~86k clean comments!?
#     * This is only possible when multiple tags are associated with each comment (eg) a comment can be classified as both toxic and obscene. 
#
# ### Multi-tagging:
# Let's check how many comments have multiple tags.

# + {"_cell_guid": "b1aa1aa1-b268-45c7-a5d6-ffa8771f15ea", "_kg_hide-input": true, "_uuid": "fb65bbdd6efc49da369578ea238dcd437e5c942b"}
x=rowsums.value_counts()

#plot
plt.figure(figsize=(8,4))
ax = sns.barplot(x.index, x.values, alpha=0.8,color=color[2])
plt.title("Multiple tags per comment")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('# of tags ', fontsize=12)

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()
# -

rowsums.value_counts()

# + {"_cell_guid": "5309efd1-5969-42a0-a361-4d509006db02", "_kg_hide-input": true, "_uuid": "d6f04ff17935dd31c256890f7eb4da7be790e98c", "_kg_hide-output": true, "cell_type": "markdown"}
# Only ~10% of the total comments have some sort of toxicity in them. There are certain comments(20) that are marked as all of the above!
#
# ## Which tags go together?
# Now let's have a look at how often the tags occur together. A good indicator of that would be a correlation plot.

# + {"_cell_guid": "f59dffec-57e1-4d82-a6c8-8d28b09d616b", "_uuid": "bf5735b2aa98d6bc87a0511a2c1d7dd18662ea0f"}
temp_df=train.iloc[:,2:-1]
# filter temp by removing clean comments
# temp_df=temp_df[~train.clean]

corr=temp_df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True)

# + {"_cell_guid": "6ef81ed4-4371-47e9-a571-350eda2b8c90", "_uuid": "297ec4fc993c2cf5c84d7bd3e931cd85872918a9", "cell_type": "markdown"}
# The above plot indicates a pattern of co-occurance but Pandas's default Corr function which uses Pearson correlation does not apply here, since the variables invovled are Categorical (binary) variables.
#
# So, to find a pattern between two categorical variables we can use other tools like 
# * Confusion matrix/Crosstab
# * Cramer's V Statistic
#     * Cramer's V stat is an extension of the chi-square test where the extent/strength of association is also measured

# + {"_cell_guid": "8f8215ed-7573-47de-bf2a-e86c253da0ee", "_kg_hide-input": true, "_uuid": "bbd11fcd7cf0f2c7ad59ab2dd3a9764cfba69e21"}
# https://pandas.pydata.org/pandas-docs/stable/style.html
def highlight_min(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_min = data == data.min()
        return [attr if v else '' for v in is_min]
    else:  # from .apply(axis=None)
        is_max = data == data.min().min()
        return pd.DataFrame(np.where(is_min, attr, ''),
                            index=data.index, columns=data.columns)

# + {"_cell_guid": "0a4a7560-71ae-4db4-85e4-c172e326505b", "_uuid": "e038525e7d14d86c7b3d294df42e9a8d626f20d1"}
#Crosstab
# Since technically a crosstab between all 6 classes is impossible to vizualize, lets take a 
# look at toxic with other tags
main_col="toxic"
corr_mats=[]
for other_col in temp_df.columns[1:]:
    confusion_matrix = pd.crosstab(temp_df[main_col], temp_df[other_col])
    corr_mats.append(confusion_matrix)
out = pd.concat(corr_mats,axis=1,keys=temp_df.columns[1:])

#cell highlighting
out = out.style.apply(highlight_min,axis=0)
out

# + {"_cell_guid": "5631221e-ce94-41b5-aace-ccb42b906310", "_uuid": "4c307d47477bcec06d2fa7cdcbd5b30e70ad3f42", "cell_type": "markdown"}
# The above table represents the Crosstab/ consufion matix of Toxic comments with the other classes. 
#
# Some interesting observations:
#
# * A Severe toxic comment is always toxic
# * Other classes seem to be a subset of toxic barring a few exceptions
#
#

# + {"_cell_guid": "4d30706c-3c0d-4bd4-b3de-de2e8f3a6486", "_kg_hide-input": false, "_uuid": "44f2a738a00e7211659a2ebbd8d33527830e925e"}
#https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix/39266194
def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

# + {"_cell_guid": "4b13fe1b-8482-49a8-bb3a-7d6976a481f2", "_uuid": "35f83d01e942b04c55ab9317f6623a09f3676f4e"}
#Checking for Toxic and Severe toxic for now
import pandas as pd
col1="toxic"
col2="severe_toxic"
confusion_matrix = pd.crosstab(temp_df[col1], temp_df[col2])
print("Confusion matrix between toxic and severe toxic:")
print(confusion_matrix)
new_corr=cramers_corrected_stat(confusion_matrix)
print("The correlation between Toxic and Severe toxic using Cramer's stat=",new_corr)

# + {"_cell_guid": "7e7c1dfc-5186-4ec2-a910-1ef3e7551ab7", "_uuid": "1fab30ff6cf3dffdec3fe723683cf42801799a67", "cell_type": "markdown"}
#
# ## Example Comments:

# + {"_cell_guid": "cf5d29a0-90b1-45cf-aa10-128a3b952306", "_kg_hide-input": true, "_uuid": "7fcfc1d19a150f7caab665cf691388e2b2f396b5"}
print("toxic:")
print(train[train.severe_toxic==1].iloc[3,1])
#print(train[train.severe_toxic==1].iloc[5,1])

# + {"_cell_guid": "797646c0-45b7-4da9-8c81-e2b9f354da6f", "_kg_hide-input": true, "_uuid": "5af966f76e9b2dadda00f760611b00f76670b45b"}
print("severe_toxic:")
print(train[train.severe_toxic==1].iloc[4,1])
#print(train[train.severe_toxic==1].iloc[4,1])

# + {"_cell_guid": "022e3edd-68c2-41dc-b299-979494b687fd", "_kg_hide-input": true, "_uuid": "e6a85250be24d8e8d2fe5e9d75e26eb3c5f9d486"}
print("Threat:")
print(train[train.threat==1].iloc[1,1])
#print(train[train.threat==1].iloc[2,1])

# + {"_cell_guid": "2616a080-5c5a-4f10-b0fa-57ee1392e707", "_kg_hide-input": true, "_uuid": "c6a4925b3ca37f74e37b9c66e35daf1aa3196776"}
print("Obscene:")
print(train[train.obscene==1].iloc[1,1])
#print(train[train.obscene==1].iloc[2,1])

# + {"_cell_guid": "27832c62-825a-422f-b114-0dfa8e684306", "_kg_hide-input": true, "_uuid": "d06773d75fdeae4a07f735d73cc8c8602c26160e"}
print("identity_hate:")
print(train[train.identity_hate==1].iloc[4,1])
#print(train[train.identity_hate==1].iloc[4,1])

# + {"_cell_guid": "dd2b2c1d-0c8a-4089-aa0f-24caa53d4ad8", "_uuid": "17a4e0e4cd4257ad4b9627ea47ae433d078e05a2", "cell_type": "markdown"}
# That was a whole lot of toxicity. Some weird observations:
#
# * Some of the comments are extremely and mere copy paste of the same thing
# * Comments can still contain IP addresses(eg:62.158.73.165), usernames(eg:ARKJEDI10) and some mystery numbers(i assume is article-IDs)
#
# Point 2 can cause huge overfitting.
#
# # Wordclouds - Frequent words:
#
# Now, let's take a look at words that are associated with these classes.
#
#    Chart Desc: The visuals here are word clouds (ie) more frequent words appear bigger. A cool way to create word clouds with funky pics is given [here](https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial). It involves the following steps.
#     * Search for an image and its base 64 encoding
#     * Paste encoding in a cell and convert it using codecs package to image
#     * Create word cloud with the new image as a mask
# A simpler way would be to create a new kaggle dataset and import images from there.
#     

# + {"_cell_guid": "76ea3ae7-e9aa-4947-ab49-39e1103bc24e", "_kg_hide-input": true, "_uuid": "e1cf9e4e0b01700d9a39f905ebce822d5eba418c", "_kg_hide-output": true}
!ls ../input/imagesforkernal/
stopword=set(STOPWORDS)

# + {"_cell_guid": "88c67243-7b91-4c4a-92fc-afd76d3f7894", "_kg_hide-input": true, "_uuid": "2fb9a7306106f8c9a74f8fbdbc5263d66eeea88a", "_kg_hide-output": false}
#clean comments
clean_mask=np.array(Image.open("../input/imagesforkernal/safe-zone.png"))
clean_mask=clean_mask[:,:,1]
#wordcloud for clean comments
subset=train[train.clean==True]
text=subset.comment_text.values
wc= WordCloud(background_color="white",max_words=2000,mask=clean_mask,stopwords=stopword)
wc.generate(" ".join(text))
plt.figure(figsize=(20,10))
plt.axis("off")
plt.title("Words frequented in Clean Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)
plt.show()
# -

text

# + {"_cell_guid": "d1483502-d0b7-464e-8fe5-2191dc5c67e9", "_kg_hide-input": true, "_uuid": "181fa5cecab3b63d82908ae52cd5d333d22750df"}
toxic_mask=np.array(Image.open("../input/imagesforkernal/toxic-sign.png"))
toxic_mask=toxic_mask[:,:,1]
#wordcloud for clean comments
subset=train[train.toxic==1]
text=subset.comment_text.values
wc= WordCloud(background_color="black",max_words=4000,mask=toxic_mask,stopwords=stopword)
wc.generate(" ".join(text))
plt.figure(figsize=(20,20))
plt.subplot(221)
plt.axis("off")
plt.title("Words frequented in Toxic Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'gist_earth' , random_state=244), alpha=0.98)

#Severely toxic comments
plt.subplot(222)
severe_toxic_mask=np.array(Image.open("../input/imagesforkernal/bomb.png"))
severe_toxic_mask=severe_toxic_mask[:,:,1]
subset=train[train.severe_toxic==1]
text=subset.comment_text.values
wc= WordCloud(background_color="black",max_words=2000,mask=severe_toxic_mask,stopwords=stopword)
wc.generate(" ".join(text))
plt.axis("off")
plt.title("Words frequented in Severe Toxic Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'Reds' , random_state=244), alpha=0.98)

#Threat comments
plt.subplot(223)
threat_mask=np.array(Image.open("../input/imagesforkernal/anger.png"))
threat_mask=threat_mask[:,:,1]
subset=train[train.threat==1]
text=subset.comment_text.values
wc= WordCloud(background_color="black",max_words=2000,mask=threat_mask,stopwords=stopword)
wc.generate(" ".join(text))
plt.axis("off")
plt.title("Words frequented in Threatening Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'summer' , random_state=2534), alpha=0.98)

#insult
plt.subplot(224)
insult_mask=np.array(Image.open("../input/imagesforkernal/swords.png"))
insult_mask=insult_mask[:,:,1]
subset=train[train.insult==1]
text=subset.comment_text.values
wc= WordCloud(background_color="black",max_words=2000,mask=insult_mask,stopwords=stopword)
wc.generate(" ".join(text))
plt.axis("off")
plt.title("Words frequented in insult Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'Paired_r' , random_state=244), alpha=0.98)

plt.show()


# + {"_cell_guid": "262e81d5-9082-4d4f-8311-f48be7388413", "_uuid": "05c18f60f9e90b3f6e6690855bb2b72085eb0d2e", "cell_type": "markdown"}
# # Feature engineering:
# I've broadly classified my feature engineering ideas into the following three groups
# ## Direct features:
# Features which are a directly due to words/content.We would be exploring the following techniques
# * Word frequency features
#     * Count features
#     * Bigrams
#     * Trigrams
# * Vector distance mapping of words (Eg: Word2Vec)
# * Sentiment scores
#
# ## Indirect features:
# Some more experimental features.
# * count of sentences 
# * count of words
# * count of unique words
# * count of letters 
# * count of punctuations
# * count of uppercase words/letters
# * count of stop words
# * Avg length of each word
#
# ## Leaky features:
# From the example, we know that the comments contain identifier information (eg: IP, username,etc.).
# We can create features out of them but, it will certainly lead to **overfitting** to this specific Wikipedia use-case.
# * toxic IP scores
# * toxic users
#
# **Note:** 
# Creating the indirect and leaky features first. There are two reasons for this,
# * Count features(Direct features) are useful only if they are created from a clean corpus
# * Also the indirect features help compensate for the loss of information when cleaning the dataset
#

# + {"_cell_guid": "791b3c4f-edbc-4bb4-8a64-763b187a459e", "_uuid": "dcff5e0f9b77be8c7f9f4b0acf173ad937e3bbdc"}
merge=pd.concat([train.iloc[:,0:2],test.iloc[:,0:2]])
df=merge.reset_index(drop=True)
# -

df.head()

# + {"_cell_guid": "ee75d2dc-4d76-445a-8fec-332c06dcf4b5", "_uuid": "0542e2a860340218dcc2a027f4bfc97f433a03ce"}
## Indirect features

#Sentense count in each comment:
    #  '\n' can be used to count the number of sentences in each comment
df['count_sent']=df["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)
#Word count in each comment:
df['count_word']=df["comment_text"].apply(lambda x: len(str(x).split()))
#Unique word count
df['count_unique_word']=df["comment_text"].apply(lambda x: len(set(str(x).split())))
#Letter count
df['count_letters']=df["comment_text"].apply(lambda x: len(str(x)))
#punctuation count
df["count_punctuations"] =df["comment_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
#upper case words count
df["count_words_upper"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
#title case words count
df["count_words_title"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
#Number of stopwords
df["count_stopwords"] = df["comment_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
#Average length of the words
df["mean_word_len"] = df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
# -

df

# + {"_cell_guid": "de492571-5990-40c2-8413-4d2da74bb72c", "_uuid": "bab9fdf2590fbbcd1db5e29480397bfb485e5e8d"}
#derived features
#Word count percent in each comment:
df['word_unique_percent']=df['count_unique_word']*100/df['count_word']
#derived features
#Punct percent in each comment:
df['punct_percent']=df['count_punctuations']*100/df['count_word']

# + {"_cell_guid": "023c6206-020f-4814-8143-de11eaa30836", "_uuid": "0d1e8136ad267b3df2692de500685f74e30c7a0d"}
#serperate train and test features
train_feats=df.iloc[0:len(train),]
test_feats=df.iloc[len(train):,]
#join the tags
train_tags=train.iloc[:,2:]
train_feats=pd.concat([train_feats,train_tags],axis=1)
# -

train_feats

# + {"_cell_guid": "41825791-11c5-4d2c-950e-fbe38ccf89f3", "_kg_hide-input": true, "_uuid": "4760e1bab3a56f3ea55d87b59359501bfe28f21b"}
train_feats['count_sent'].loc[train_feats['count_sent']>10] = 10 
plt.figure(figsize=(12,6))
## sentenses
plt.subplot(121)
plt.suptitle("Are longer comments more toxic?",fontsize=20)
sns.violinplot(y='count_sent',x='clean', data=train_feats,split=True)
plt.xlabel('Clean?', fontsize=12)
plt.ylabel('# of sentences', fontsize=12)
plt.title("Number of sentences in each comment", fontsize=15)
# words
train_feats['count_word'].loc[train_feats['count_word']>200] = 200
plt.subplot(122)
sns.violinplot(y='count_word',x='clean', data=train_feats,split=True,inner="quart")
plt.xlabel('Clean?', fontsize=12)
plt.ylabel('# of words', fontsize=12)
plt.title("Number of words in each comment", fontsize=15)

plt.show()

# + {"_cell_guid": "64cefc8c-0a27-4e30-ab28-8941c962208f", "_uuid": "899d447ba09f47c73130c4aaee08f842cb034346", "cell_type": "markdown"}
# Long sentences or more words do not seem to be a significant indicator of toxicity.
#
# Chart desc: Violin plot is an alternative to the traditional box plot. The inner markings show the percentiles while the width of the "violin" shows the volume of comments at that level/instance.

# + {"_cell_guid": "680419e0-2447-48b3-8b78-d0a537cec24c", "_uuid": "937e48180a8a554837927c07edf3561af0e45ac2"}
train_feats['count_unique_word'].loc[train_feats['count_unique_word']>200] = 200
#prep for split violin plots
#For the desired plots , the data must be in long format
temp_df = pd.melt(train_feats, value_vars=['count_word', 'count_unique_word'], id_vars='clean')
#spammers - comments with less than 40% unique words
spammers=train_feats[train_feats['word_unique_percent']<30]

# + {"_cell_guid": "daf5ed70-88f0-43b8-b76e-a76cfe076194", "_uuid": "961b30aae2ed652e39ede57ef6193d65a97b492d"}
plt.figure(figsize=(16,12))
plt.suptitle("What's so unique ?",fontsize=20)
gridspec.GridSpec(2,2)
plt.subplot2grid((2,2),(0,0))
sns.violinplot(x='variable', y='value', hue='clean', data=temp_df,split=True,inner='quartile')
plt.title("Absolute wordcount and unique words count")
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.subplot2grid((2,2),(0,1))
plt.title("Percentage of unique words of total words in comment")
#sns.boxplot(x='clean', y='word_unique_percent', data=train_feats)
ax=sns.kdeplot(train_feats[train_feats.clean == 0].word_unique_percent, label="Bad",shade=True,color='r')
ax=sns.kdeplot(train_feats[train_feats.clean == 1].word_unique_percent, label="Clean")
plt.legend()
plt.ylabel('Number of occurances', fontsize=12)
plt.xlabel('Percent unique words', fontsize=12)

x=spammers.iloc[:,-7:].sum()
plt.subplot2grid((2,2),(1,0),colspan=2)
plt.title("Count of comments with low(<30%) unique words",fontsize=15)
ax=sns.barplot(x=x.index, y=x.values,color=color[3])

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.xlabel('Threat class', fontsize=12)
plt.ylabel('# of comments', fontsize=12)
plt.show()

# + {"_cell_guid": "0e78458e-5d3e-43ce-ab9b-552473e46073", "_uuid": "fd3b009c2101355822808b66fda83d7afd595b47", "cell_type": "markdown"}
# ### Word count VS unique word count:
# There are noticeable shifts in the mean of both word count and unique word count across clean and toxic comments.
#    * Chart desc: The first chart is a split violin chart. It is a variation of the traditional box chart/violin chart which allows us to split the violin in the middle based on a categorical variable.
#    
# ### Unique word count percent:
# There is a bulge near the 0-10% mark which indicates a large number of toxic comments which contain very little variety of words.
#    * Chart desc: The second chart is an overlay of two kernel density estimation plots of percentage of unique words out of all the words in the comment, done for both clean and toxic comments
#
# Even though the number of clean comments dominates the dataset(~90%), there are only 75 clean comments that are spam, which makes it a powerful indicator of a toxic comment.
# # Spammers are more toxic!
# No surprises here. Let's take a look at some clean and toxic spam messages

# + {"_cell_guid": "0cdca874-a723-4129-b5e4-2487686742e6", "_uuid": "70f865629614edda911dc89d6882afd752df117c"}
print("Clean Spam example:")
print(spammers[spammers.clean==1].comment_text.iloc[1])
print("Toxic Spam example:")
print(spammers[spammers.toxic==1].comment_text.iloc[2])

# + {"_cell_guid": "2473e0d6-c225-4795-b3a0-30ff3d36e9fe", "_uuid": "a078906de353ef7f8f41ff13c3d75881f8c5b1ee", "cell_type": "markdown"}
# # Spam is toxic to the model too!
#
# These spam entries are bad if we design our model to contain normal word counts features.
# Imagine the scenario in which our model picked up the words "mitt romney" from any comment and classified it as toxic :(
#
# -

# CountVectorizer Example
# https://qr.ae/TWnHgJ
text1 = "Jon Skinner left his job as a software engineer at Google in order to pursue a dream: to build a better text editor. As you can see below, the first glimpse of Sublime Text that he gave to the public, back in 2007, looks eerily similar to the version we are using today. In his blogs he mentions three guiding principles he had in mind while developing Sublime Text"
txt1_sent =nltk.sent_tokenize(text1)
print(txt1_sent)

# +
# Fit to sentence
cv = CountVectorizer().fit(txt1_sent)

# Print feature names
print(cv.get_feature_names())

# Vocab
print('\n')
print('Vocabulary')
print(cv.vocabulary_)

# Transform
text_vect = cv.transform(txt1_sent)

print("\n")

# To array
print("Array")
print(text_vect.toarray())

# + {"_cell_guid": "b1b006df-6229-4a32-9d68-112e6dee72a5", "_uuid": "2108f5575a7a16a528d80eb97d5c6a2ed0c10f79", "cell_type": "markdown"}
# # Leaky features
# **Caution:** Even-though including these features might help us perform better in this particular scenario, it will not make sence to add them in the final model/general purpose model.
#
# Here we are creating our own custom count vectorizer to create count variables that match our regex condition.
#

# + {"_cell_guid": "0a32bebf-653c-4549-ab20-999bb2a04b74", "_uuid": "25434ef58124c131e57f88626d3ad5ed6a6d22ab"}
#Leaky features
df['ip']=df["comment_text"].apply(lambda x: re.findall("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",str(x)))
#count of ip addresses
df['count_ip']=df["ip"].apply(lambda x: len(x))

#links
df['link']=df["comment_text"].apply(lambda x: re.findall("http://.*com",str(x)))
#count of links
df['count_links']=df["link"].apply(lambda x: len(x))

#article ids
df['article_id']=df["comment_text"].apply(lambda x: re.findall("\d:\d\d\s{0,5}$",str(x)))
df['article_id_flag']=df.article_id.apply(lambda x: len(x))

#username
##              regex for     Match anything with [[User: ---------- ]]
# regexp = re.compile("\[\[User:(.*)\|")
df['username']=df["comment_text"].apply(lambda x: re.findall("\[\[User(.*)\|",str(x)))
#count of username mentions
df['count_usernames']=df["username"].apply(lambda x: len(x))
#check if features are created
#df.username[df.count_usernames>0]

# Leaky Ip
cv = CountVectorizer()
count_feats_ip = cv.fit_transform(df["ip"].apply(lambda x : str(x)))


# Leaky usernames

cv = CountVectorizer()
count_feats_user = cv.fit_transform(df["username"].apply(lambda x : str(x)))



# + {"_cell_guid": "d47b21ba-9697-44af-9ad9-403541d1f108", "_uuid": "ab0f81d44c939a02703757d9e54bbeceeeb58b37"}
df[df.count_usernames!=0].comment_text.iloc[0]

# + {"_cell_guid": "5634dd63-b5ef-482a-974b-d26adef3c3c7", "_uuid": "eee465721ef0de53399861a3118cbea9de2084b6"}
# check few names
cv.get_feature_names()[120:130]

# + {"_cell_guid": "5aa04054-04f3-48ed-be6d-7f4a8e40c885", "_uuid": "5a3ecf31a965297e20b060a5992ec84f3b546dcc", "cell_type": "markdown"}
# # Leaky Feature Stability:
# Checking the re-occurance of leaky features to check their utility in predicting the test set. 
#
# [Discussion on leaky feature stability](https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda#263577)
#
#

# + {"_cell_guid": "c86bbaba-f7c1-414b-b943-76340dc50a60", "_uuid": "63a1e95e951ae76fb55871fb7a6dbf4e22a9682d"}
leaky_feats=df[["ip","link","article_id","username","count_ip","count_links","count_usernames","article_id_flag"]]
leaky_feats_train=leaky_feats.iloc[:train.shape[0]]
leaky_feats_test=leaky_feats.iloc[train.shape[0]:]

# + {"_cell_guid": "2711e8d5-cfae-402e-906a-f677848018cc", "_uuid": "ac17caea13d780254ac967b7f3c3417cf1a7778b"}
#filterout the entries without ips
train_ips=leaky_feats_train.ip[leaky_feats_train.count_ip!=0]
test_ips=leaky_feats_test.ip[leaky_feats_test.count_ip!=0]
#get the unique list of ips in test and train datasets
train_ip_list=list(set([a for b in train_ips.tolist() for a in b]))
test_ip_list=list(set([a for b in test_ips.tolist() for a in b]))

# get common elements
common_ip_list=list(set(train_ip_list).intersection(test_ip_list))
plt.title("Common IP addresses")
venn.venn2(subsets=(len(train_ip_list),len(test_ip_list),len(common_ip_list)),set_labels=("# of unique IP in train","# of unique IP in test"))
plt.show()
# -

# Flatten an array list
[a for c in train_ips for a in c]

# + {"_cell_guid": "13accd38-3a6c-4da3-9257-324ec6d3f91e", "_uuid": "17e793e5306063bc11d60a23c9f0340027798178"}
#filterout the entries without links
train_links=leaky_feats_train.link[leaky_feats_train.count_links!=0]
test_links=leaky_feats_test.link[leaky_feats_test.count_links!=0]
#get the unique list of ips in test and train datasets
train_links_list=list(set([a for b in train_links.tolist() for a in b]))
test_links_list=list(set([a for b in test_links.tolist() for a in b]))

# get common elements
common_links_list=list(set(train_links_list).intersection(test_links_list))
plt.title("Common links")
venn.venn2(subsets=(len(train_links_list),len(test_links_list),len(common_links_list)),
           set_labels=("# of unique links in train","# of unique links in test"))
plt.show()

# + {"_cell_guid": "feed1227-9b93-4376-a326-acd79a6e0281", "_uuid": "cdab51e8a5565dd5e95197018a9b67a53c4b4940"}
#filterout the entries without users
train_users=leaky_feats_train.username[leaky_feats_train.count_usernames!=0]
test_users=leaky_feats_test.username[leaky_feats_test.count_usernames!=0]
#get the unique list of ips in test and train datasets
train_users_list=list(set([a for b in train_users.tolist() for a in b]))
test_users_list=list(set([a for b in test_users.tolist() for a in b]))

# get common elements
common_users_list=list(set(train_users_list).intersection(test_users_list))
plt.title("Common usernames")
venn.venn2(subsets=(len(train_users_list),len(test_users_list),len(common_users_list)),
           set_labels=("# of unique Usernames in train","# of unique usernames in test"))
plt.show()

# + {"_cell_guid": "6a9d791e-a40b-43a7-a049-2ae6822ebb86", "_uuid": "46daf7b9ae369b393281ba5b9213120982f799f9", "cell_type": "markdown"}
# The feature stability (aka) the reoccurance of train dataset usernames in the test dataset seems to be minimal. 
# We can just use the intersection (eg) the common IPs/links for test and train in our feature engineering.
#
# Another usecase for the list of IPs would be to find out if they are a part of the [blocked IP list](https://en.wikipedia.org/wiki/Wikipedia:Database_reports/Indefinitely_blocked_IPs)

# + {"_cell_guid": "3293defd-f8e8-4584-a237-5fb965ebf550", "_kg_hide-input": true, "_uuid": "ee2825b3889a62c64e2a74bc48aeab0d5d922cda"}
#https://en.wikipedia.org/wiki/Wikipedia:Database_reports/Indefinitely_blocked_IPs)

blocked_ips=["216.102.6.176",
"216.120.176.2",
"203.25.150.5",
"203.217.8.30",
"66.90.101.58",
"125.178.86.75",
"210.15.217.194",
"69.36.166.207",
"213.25.24.253",
"24.60.181.235",
"71.204.14.32",
"216.91.92.18",
"212.219.2.4",
"194.74.190.162",
"64.15.152.246",
"59.100.76.166",
"146.145.221.129",
"146.145.221.130",
"74.52.44.34",
"68.5.96.201",
"65.184.176.45",
"209.244.43.209",
"82.46.9.168",
"209.200.236.32",
"209.200.229.181",
"202.181.99.22",
"220.233.226.170",
"212.138.64.178",
"220.233.227.249",
"72.14.194.31",
"72.249.45.0/24",
"72.249.44.0/24",
"80.175.39.213",
"81.109.164.45",
"64.157.15.0/24",
"208.101.10.54",
"216.157.200.254",
"72.14.192.14",
"204.122.16.13",
"217.156.39.245",
"210.11.188.16",
"210.11.188.17",
"210.11.188.18",
"210.11.188.19",
"210.11.188.20",
"64.34.27.153",
"209.68.139.150",
"152.163.100.0/24",
"65.175.48.2",
"131.137.245.197",
"131.137.245.199",
"131.137.245.200",
"64.233.172.37",
"66.99.182.25",
"67.43.21.12",
"66.249.85.85",
"65.175.134.11",
"201.218.3.198",
"193.213.85.12",
"131.137.245.198",
"83.138.189.74",
"72.14.193.163",
"66.249.84.69",
"209.204.71.2",
"80.217.153.189",
"83.138.136.92",
"83.138.136.91",
"83.138.189.75",
"83.138.189.76",
"212.100.250.226",
"212.100.250.225",
"212.159.98.189",
"87.242.116.201",
"74.53.243.18",
"213.219.59.96/27",
"212.219.82.37",
"203.38.149.226",
"66.90.104.22",
"125.16.137.130",
"66.98.128.0/17",
"217.33.236.2",
"24.24.200.113",
"152.22.0.254",
"59.145.89.17",
"71.127.224.0/20",
"65.31.98.71",
"67.53.130.69",
"204.130.130.0/24",
"72.14.193.164",
"65.197.143.214",
"202.60.95.235",
"69.39.89.95",
"88.80.215.14",
"216.218.214.2",
"81.105.175.201",
"203.108.239.12",
"74.220.207.168",
"206.253.55.206",
"206.253.55.207",
"206.253.55.208",
"206.253.55.209",
"206.253.55.210",
"66.64.56.194",
"70.91.90.226",
"209.60.205.96",
"202.173.191.210",
"169.241.10.83",
"91.121.195.205",
"216.70.136.88",
"72.228.151.208",
"66.197.167.120",
"212.219.232.81",
"208.86.225.40",
"63.232.20.2",
"206.219.189.8",
"212.219.14.0/24",
"165.228.71.6",
"99.230.151.129",
"72.91.11.99",
"173.162.177.53",
"60.242.166.182",
"212.219.177.34",
"12.104.27.5",
"85.17.92.13",
"91.198.174.192/27",
"155.246.98.61",
"71.244.123.63",
"81.144.152.130",
"198.135.70.1",
"71.255.126.146",
"74.180.82.59",
"206.158.2.80",
"64.251.53.34",
"24.29.92.238",
"76.254.235.105",
"68.96.242.239",
"203.202.234.226",
"173.72.89.88",
"87.82.229.195",
"68.153.245.37",
"216.240.128.0/19",
"72.46.129.44",
"66.91.35.165",
"82.71.49.124",
"69.132.171.231",
"75.145.183.129",
"194.80.20.237",
"98.207.253.170",
"76.16.222.162",
"66.30.100.130",
"96.22.29.23",
"76.168.140.158",
"202.131.166.252",
"89.207.212.99",
"81.169.155.246",
"216.56.8.66",
"206.15.235.10",
"115.113.95.20",
"204.209.59.11",
"27.33.141.67",
"41.4.65.162",
"99.6.65.6",
"60.234.239.169",
"2620:0:862:101:0:0:2:0/124",
"183.192.165.31",
"50.68.6.12",
"37.214.82.134",
"96.50.0.230",
"60.231.28.109",
"64.90.240.50",
"49.176.97.12",
"209.80.150.137",
"24.22.67.116",
"206.180.81.2",
"195.194.39.100",
"87.41.52.6",
"169.204.164.227",
"50.137.55.117",
"50.77.84.161",
"90.202.230.247",
"186.88.129.224",
"2A02:EC80:101:0:0:0:2:0/124",
"142.4.117.177",
"86.40.105.198",
"120.43.20.149",
"198.199.64.0/18",
"192.34.56.0/21",
"192.81.208.0/20",
"2604:A880:0:0:0:0:0:0/32",
"108.72.107.229",
"2602:306:CC2B:7000:41D3:B92D:731C:959D",
"185.15.59.201",
"180.149.1.229",
"207.191.188.66",
"210.22.63.92",
"117.253.196.217",
"119.160.119.172",
"90.217.133.223",
"194.83.8.3",
"194.83.164.22",
"217.23.228.149",
"65.18.58.1",
"168.11.15.2",
"65.182.127.31",
"207.106.153.252",
"64.193.88.2",
"152.26.71.2",
"199.185.67.179",
"117.90.240.73",
"108.176.58.170",
"195.54.40.28",
"185.35.164.109",
"192.185.0.0/16",
"2605:E000:1605:C0C0:3D3D:A148:3039:71F1",
"107.158.0.0/16",
"85.159.232.0/21",
"69.235.4.10",
"86.176.166.206",
"108.65.152.51",
"10.4.1.0/24",
"103.27.227.139",
"188.55.31.191",
"188.53.13.34",
"176.45.58.252",
"176.45.22.37",
"24.251.44.140",
"108.200.140.191",
"117.177.169.4",
"72.22.162.38",
"24.106.242.82",
"79.125.190.93",
"107.178.200.1",
"123.16.244.246",
"83.228.167.87",
"128.178.197.53",
"14.139.172.18",
"207.108.136.254",
"184.152.17.217",
"186.94.29.73",
"217.200.199.2",
"66.58.141.104",
"166.182.81.30",
"89.168.206.116",
"92.98.163.145",
"77.115.31.71",
"178.36.118.74",
"157.159.10.14",
"103.5.212.139",
"203.174.180.226",
"69.123.252.95",
"199.200.123.233",
"121.45.89.82",
"71.228.87.155",
"68.189.67.92",
"216.161.176.152",
"98.17.30.139",
"2600:1006:B124:84BD:0:0:0:103",
"117.161.0.0/16",
"12.166.68.34",
"96.243.149.64",
"74.143.90.218",
"76.10.176.221",
"104.250.128.0/19",
"185.22.183.128/25",
"89.105.194.64/26",
"202.45.119.0/24",
"73.9.140.64",
"164.127.71.72",
"50.160.129.2",
"49.15.213.207",
"83.7.192.0/18",
"201.174.63.79",
"2A02:C7D:4643:8F00:D09D:BE1:D2DE:BB1F",
"125.60.195.230",
"49.145.113.145",
"168.18.160.134",
"72.193.218.222",
"199.216.164.10",
"120.144.130.89",
"104.130.67.208",
"50.160.221.147",
"163.47.141.50",
"91.200.12.136",
"83.222.0.0/19",
"67.231.16.0/20",
"72.231.0.196",
"180.216.68.197",
"183.160.178.135",
"183.160.176.16",
"24.25.221.150",
"92.222.109.43",
"142.134.243.215",
"216.181.221.72",
"113.205.170.110",
"74.142.2.98",
"192.235.8.3",
"2402:4000:BBFC:36FC:E469:F2F0:9351:71A0",
"80.244.81.191",
"2607:FB90:1377:F765:D45D:46BF:81EA:9773",
"2600:1009:B012:7D88:418B:54BA:FCBC:4584",
"104.237.224.0/19",
"2600:1008:B01B:E495:C05A:7DD3:926:E83C",
"168.8.249.234",
"162.211.179.36",
"138.68.0.0/16",
"145.236.37.195",
"67.205.128.0/18",
"2A02:C7D:2832:CE00:B914:19D6:948D:B37D",
"107.77.203.212",
"2607:FB90:65C:A136:D46F:23BA:87C2:3D10",
"2A02:C7F:DE2F:7900:5D64:E991:FFF0:FA93",
"82.23.32.186",
"106.76.243.74",
"82.33.48.223",
"180.216.160.0/19",
"94.102.184.35",
"94.102.184.26",
"109.92.162.54",
"2600:8800:7180:BF00:4C27:4591:347C:736C",
"178.41.186.50",
"184.97.134.128",
"176.221.32.0/22",
"207.99.40.142",
"109.97.241.134",
"82.136.64.19",
"91.236.74.119",
"197.210.0.0/16",
"173.230.128.0/19",
"162.216.16.0/22",
"80.111.222.211",
"191.37.28.21",
"124.124.103.194",
"50.207.7.198",
"220.233.131.98",
"107.77.241.11",
"68.112.39.0/27",
"173.236.128.0/17",
"49.49.240.24",
"96.31.10.178",
"50.251.229.75"]


# + {"_cell_guid": "424ce78f-d0fc-444a-a52e-fdf563d193cb", "_uuid": "306baf0a16900a0ec8d304d5ed636950d742f52f"}
train_ip_list=list(set([a for b in train_ips.tolist() for a in b]))
test_ip_list=list(set([a for b in test_ips.tolist() for a in b]))

# get common elements
blocked_ip_list_train=list(set(train_ip_list).intersection(blocked_ips))
blocked_ip_list_test=list(set(test_ip_list).intersection(blocked_ips))

print("There are",len(blocked_ip_list_train),"blocked IPs in train dataset")
print("There are",len(blocked_ip_list_test),"blocked IPs in test dataset")

# + {"_cell_guid": "e2f08966-f2d2-41a8-9d5a-0947271f3751", "_uuid": "3f063125dd70fef845e6c42905e6113c932bc301", "cell_type": "markdown"}
# An interesting but somewhat insignificant finding. There are 6 blocked IP mentions in the comments overall. 
#
# Anyways, moving on to cleaning the dataset.

# + {"_cell_guid": "a2c9fd43-d973-4729-9410-4ffb9d5c7d85", "_uuid": "89f23d9612f9a6a9208dfc7672ce7e08df5624db"}
end_time=time.time()
print("total time till Leaky feats",end_time-start_time)

# + {"_cell_guid": "6ede73e4-8f08-4a24-8616-138ad6c9e771", "_uuid": "4d56a85191de5bd335e99d045c0b94315683dfa5", "cell_type": "markdown"}
# # Corpus cleaning:
#
# Its important to use a clean dataset before creating count features. 

# + {"_cell_guid": "18ecbb15-dfad-4263-9d54-9b335d866495", "_uuid": "7f93aed2e5fb005871b4b302681411cdfc4d7ede"}
corpus=merge.comment_text

# + {"_cell_guid": "a2718f9a-ab5e-43c9-a807-62dfd136653e", "_kg_hide-output": true, "_kg_hide-input": true, "_uuid": "512d130b7a29afc61a60aa22769df7f72e3b1995"}
#https://drive.google.com/file/d/0B1yuv8YaUVlZZ1RzMFJmc1ZsQmM/view
# Aphost lookup dict
APPO = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"
}

# + {"_cell_guid": "a318c92f-75e1-4822-a438-462133d6a9bf", "_uuid": "d43a9ceeb372635dad0c1d6d5c78ebc671c00597"}
def clean(comment):
    """
    This function receives comments and returns clean word-list
    """
    #Convert to lower case , so that Hi and hi are the same
    comment=comment.lower()
    #remove \n
    comment=re.sub("\\n","",comment)
    # remove leaky elements like ip,user
    comment=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",comment)
    #removing usernames
    comment=re.sub("\[\[.*\]","",comment)
    
    #Split the sentences into words
    words=tokenizer.tokenize(comment)
    
    # (')aphostophe  replacement (ie)   you're --> you are  
    # ( basic dictionary lookup : master dictionary present in a hidden block of code)
    words=[APPO[word] if word in APPO else word for word in words]
    words=[lem.lemmatize(word, "v") for word in words]
    words = [w for w in words if not w in eng_stopwords]
    
    clean_sent=" ".join(words)
    # remove any non alphanum,digit character
    #clean_sent=re.sub("\W+"," ",clean_sent)
    #clean_sent=re.sub("  "," ",clean_sent)
    return(clean_sent)

# + {"_cell_guid": "01622089-ad76-495a-91fe-4ee6419ab323", "_uuid": "31c43f18a851a470f0de6e4774fa62c66030dbae"}
corpus.iloc[12230]

# + {"_cell_guid": "120bf502-26fa-4fd4-8653-1c0c8115063d", "_uuid": "7635ceb7bd20b714e128878cef878affda9e3e39"}
clean(corpus.iloc[12230])

# + {"_cell_guid": "500c7b8b-fdd2-4363-b054-7e0bc5a2aa49", "_uuid": "55417e63e1bc7588085f2aabc186b02d748a59c4"}
clean_corpus=corpus.apply(lambda x :clean(x))

end_time=time.time()
print("total time till Cleaning",end_time-start_time)

# + {"_cell_guid": "f79356b1-ea18-4afd-9c70-7b058189108a", "_kg_hide-output": true, "_kg_hide-input": true, "_uuid": "430cc56e4c1b10ac1ee1b63ff8a7cd1c1d936a02"}
# To do next:
# Slang lookup dictionary for sentiments
#http://slangsd.com/data/SlangSD.zip
#http://arxiv.org/abs/1608.05129
# dict lookup 
#https://bytes.com/topic/python/answers/694819-regular-expression-dictionary-key-search


# + {"_cell_guid": "90055d89-d381-4c1e-b7e9-820660c2eb52", "_uuid": "cd5c9877d1443bfd3379b8f038c44af8cd5189fa", "cell_type": "markdown"}
# # Direct features:
#
# ## 1)Count based features(for unigrams):
# Lets create some features based on frequency distribution of the words. Initially lets consider taking words one at a time (ie) Unigrams
#
# Python's SKlearn provides 3 ways of creating count features.All three of them first create a vocabulary(dictionary) of words and then create a [sparse matrix](#https://en.wikipedia.org/wiki/Sparse_matrix) of word counts for the words in the sentence that are present in the dictionary. A brief description of them:
# * CountVectorizer
#     * Creates a matrix with frequency counts of each word in the text corpus
# * TF-IDF Vectorizer
#     * TF - Term Frequency -- Count of the words(Terms) in the text corpus (same of Count Vect)
#     * IDF - Inverse Document Frequency -- Penalizes words that are too frequent. We can think of this as regularization
# * HashingVectorizer
#     * Creates a hashmap(word to number mapping based on hashing technique) instead of a dictionary for vocabulary
#     * This enables it to be more scalable and faster for larger text coprus
#     * Can be parallelized across multiple threads
#         
# Using TF-IDF here.
# Note: Using the concatenated dataframe "merge" which contains both text from train and test dataset to ensure that the vocabulary that we create does not missout on the words that are unique to testset.

# + {"_cell_guid": "69cab7b5-0e40-4705-94e4-c68051ca5c25", "_kg_hide-input": true, "_uuid": "705d3d209428c1e0154645c802c74fdd6ec052c8"}
### Unigrams -- TF-IDF 
# using settings recommended here for TF-IDF -- https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle

#some detailed description of the parameters
# min_df=10 --- ignore terms that appear lesser than 10 times 
# max_features=None  --- Create as many words as present in the text corpus
    # changing max_features to 10k for memmory issues
# analyzer='word'  --- Create features from words (alternatively char can also be used)
# ngram_range=(1,1)  --- Use only one word at a time (unigrams)
# strip_accents='unicode' -- removes accents
# use_idf=1,smooth_idf=1 --- enable IDF
# sublinear_tf=1   --- Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf)


#temp settings to min=200 to facilitate top features section to run in kernals
#change back to min=10 to get better results
start_unigrams=time.time()
tfv = TfidfVectorizer(min_df=200,  max_features=10000, 
            strip_accents='unicode', analyzer='word',ngram_range=(1,1),
            use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
tfv.fit(clean_corpus)
features = np.array(tfv.get_feature_names())

train_unigrams =  tfv.transform(clean_corpus.iloc[:train.shape[0]])
test_unigrams = tfv.transform(clean_corpus.iloc[train.shape[0]:])
# -

# Features
features

# Data
train_unigrams.toarray()

# + {"_cell_guid": "e0caea47-da84-4be3-9a39-e36f6a0de8d9", "_kg_hide-output": true, "_kg_hide-input": true, "_uuid": "4464c37c1ca84a60735e255ee0a8fa3858e948e6"}
#https://buhrmann.github.io/tfidf-analysis.html
def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    
    D = Xtr[grp_ids].toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

# modified for multilabel milticlass
def top_feats_by_class(Xtr, features, min_tfidf=0.1, top_n=20):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    cols=train_tags.columns
    for col in cols:
        ids = train_tags.index[train_tags[col]==1]
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

# + {"_cell_guid": "64474aa3-a5b9-47c9-898d-111f85ff99b5", "_uuid": "b25110d443b1e78ae5ee57c48acf116e824ec49a"}
#get top n for unigrams
tfidf_top_n_per_lass=top_feats_by_class(train_unigrams,features)

end_unigrams=time.time()

print("total time in unigrams",end_unigrams-start_unigrams)
print("total time till unigrams",end_unigrams-start_time)

# -

print(train_tags.columns)
tfidf_top_n_per_lass[6]

# + {"_cell_guid": "f72270c2-fd1a-4d55-9e9f-36b55ebc10fd", "_kg_hide-input": true, "_uuid": "8f424101c06cfae85add9517442ced9607e09a1e"}
plt.figure(figsize=(16,22))
plt.suptitle("TF_IDF Top words per class(unigrams)",fontsize=20)
gridspec.GridSpec(4,2)
plt.subplot2grid((4,2),(0,0))
sns.barplot(tfidf_top_n_per_lass[0].feature.iloc[0:9],tfidf_top_n_per_lass[0].tfidf.iloc[0:9],color=color[0])
plt.title("class : Toxic",fontsize=15)
plt.xlabel('Word', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)

plt.subplot2grid((4,2),(0,1))
sns.barplot(tfidf_top_n_per_lass[1].feature.iloc[0:9],tfidf_top_n_per_lass[1].tfidf.iloc[0:9],color=color[1])
plt.title("class : Severe toxic",fontsize=15)
plt.xlabel('Word', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)


plt.subplot2grid((4,2),(1,0))
sns.barplot(tfidf_top_n_per_lass[2].feature.iloc[0:9],tfidf_top_n_per_lass[2].tfidf.iloc[0:9],color=color[2])
plt.title("class : Obscene",fontsize=15)
plt.xlabel('Word', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)


plt.subplot2grid((4,2),(1,1))
sns.barplot(tfidf_top_n_per_lass[3].feature.iloc[0:9],tfidf_top_n_per_lass[3].tfidf.iloc[0:9],color=color[3])
plt.title("class : Threat",fontsize=15)
plt.xlabel('Word', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)


plt.subplot2grid((4,2),(2,0))
sns.barplot(tfidf_top_n_per_lass[4].feature.iloc[0:9],tfidf_top_n_per_lass[4].tfidf.iloc[0:9],color=color[4])
plt.title("class : Insult",fontsize=15)
plt.xlabel('Word', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)


plt.subplot2grid((4,2),(2,1))
sns.barplot(tfidf_top_n_per_lass[5].feature.iloc[0:9],tfidf_top_n_per_lass[5].tfidf.iloc[0:9],color=color[5])
plt.title("class : Identity hate",fontsize=15)
plt.xlabel('Word', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)


plt.subplot2grid((4,2),(3,0),colspan=2)
sns.barplot(tfidf_top_n_per_lass[6].feature.iloc[0:19],tfidf_top_n_per_lass[6].tfidf.iloc[0:19])
plt.title("class : Clean",fontsize=15)
plt.xlabel('Word', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)

plt.show()

# + {"_cell_guid": "d7545663-00e9-4fd3-8def-986e38826133", "_kg_hide-input": false, "_uuid": "7cfe0e97eb58370ee9f43aae9a0c3739b2506224"}
#temp settings to min=150 to facilitate top features section to run in kernals
#change back to min=10 to get better results
tfv = TfidfVectorizer(min_df=150,  max_features=30000, 
            strip_accents='unicode', analyzer='word',ngram_range=(2,2),
            use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

tfv.fit(clean_corpus)
features = np.array(tfv.get_feature_names())
train_bigrams =  tfv.transform(clean_corpus.iloc[:train.shape[0]])
test_bigrams = tfv.transform(clean_corpus.iloc[train.shape[0]:])
#get top n for bigrams
tfidf_top_n_per_lass=top_feats_by_class(train_bigrams,features)

# + {"_cell_guid": "3108a776-e49b-4f17-8ef1-8dd41d2f25f2", "_kg_hide-input": true, "_uuid": "bec45d270989455db1ca21b0b628e8001acb0dd7"}
plt.figure(figsize=(16,22))
plt.suptitle("TF_IDF Top words per class(Bigrams)",fontsize=20)
gridspec.GridSpec(4,2)
plt.subplot2grid((4,2),(0,0))
sns.barplot(tfidf_top_n_per_lass[0].feature.iloc[0:5],tfidf_top_n_per_lass[0].tfidf.iloc[0:5],color=color[0])
plt.title("class : Toxic",fontsize=15)
plt.xlabel('Word', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)

plt.subplot2grid((4,2),(0,1))
sns.barplot(tfidf_top_n_per_lass[1].feature.iloc[0:5],tfidf_top_n_per_lass[1].tfidf.iloc[0:5],color=color[1])
plt.title("class : Severe toxic",fontsize=15)
plt.xlabel('Word', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)


plt.subplot2grid((4,2),(1,0))
sns.barplot(tfidf_top_n_per_lass[2].feature.iloc[0:5],tfidf_top_n_per_lass[2].tfidf.iloc[0:5],color=color[2])
plt.title("class : Obscene",fontsize=15)
plt.xlabel('Word', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)


plt.subplot2grid((4,2),(1,1))
sns.barplot(tfidf_top_n_per_lass[3].feature.iloc[0:5],tfidf_top_n_per_lass[3].tfidf.iloc[0:5],color=color[3])
plt.title("class : Threat",fontsize=15)
plt.xlabel('Word', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)


plt.subplot2grid((4,2),(2,0))
sns.barplot(tfidf_top_n_per_lass[4].feature.iloc[0:5],tfidf_top_n_per_lass[4].tfidf.iloc[0:5],color=color[4])
plt.title("class : Insult",fontsize=15)
plt.xlabel('Word', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)


plt.subplot2grid((4,2),(2,1))
sns.barplot(tfidf_top_n_per_lass[5].feature.iloc[0:5],tfidf_top_n_per_lass[5].tfidf.iloc[0:5],color=color[5])
plt.title("class : Identity hate",fontsize=15)
plt.xlabel('Word', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)


plt.subplot2grid((4,2),(3,0),colspan=2)
sns.barplot(tfidf_top_n_per_lass[6].feature.iloc[0:9],tfidf_top_n_per_lass[6].tfidf.iloc[0:9])
plt.title("class : Clean",fontsize=15)
plt.xlabel('Word', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)

plt.show()

# + {"_cell_guid": "370067c2-4af8-458d-bc2c-4ddae025047b", "_uuid": "1a505e815eb3aecf9be4fbea2c9362f57f49dfe6"}
end_time=time.time()
print("total time till bigrams",end_time-start_time)

# + {"_cell_guid": "9cba12bd-451a-41b4-932b-5470b84bf715", "_uuid": "e9d43cbeb2495307f6ffc611a70cdb365940b48f"}
tfv = TfidfVectorizer(min_df=100,  max_features=30000, 
            strip_accents='unicode', analyzer='char',ngram_range=(1,4),
            use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

tfv.fit(clean_corpus)
features = np.array(tfv.get_feature_names())
train_charngrams =  tfv.transform(clean_corpus.iloc[:train.shape[0]])
test_charngrams = tfv.transform(clean_corpus.iloc[train.shape[0]:])
end_time=time.time()
print("total time till charngrams",end_time-start_time)
# -

tfv.get_feature_names()

# + {"_cell_guid": "3ac374cb-f099-4889-bfb0-8c1fd7112c18", "_uuid": "38bd9fcbb54b50b2e80b0ff38b60cfa09d30230b", "cell_type": "markdown"}
# # Baseline Model:

# + {"_cell_guid": "df7bf89a-3db4-47b9-92e3-e3be02a05901", "_uuid": "7e671416f5159306be58c0c3a31a08ca10b8a6ae"}
#Credis to AlexSanchez https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline-eda-0-052-lb#261316
#custom NB model

class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)
        return self
    
# model = NbSvmClassifier(C=4, dual=True, n_jobs=-1)


# + {"_cell_guid": "4887bc26-d793-4257-99cb-65876cf3d77a", "_uuid": "d780a4681339cffec2724a6a037df4baf58fd365"}
SELECTED_COLS=['count_sent', 'count_word', 'count_unique_word',
       'count_letters', 'count_punctuations', 'count_words_upper',
       'count_words_title', 'count_stopwords', 'mean_word_len',
       'word_unique_percent', 'punct_percent']
target_x=train_feats[SELECTED_COLS]
# target_x

TARGET_COLS=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
target_y=train_tags[TARGET_COLS]

# Strat k fold due to imbalanced classes
# split = StratifiedKFold(n_splits=2, random_state=1)

#https://www.kaggle.com/yekenot/toxic-regression

# + {"_cell_guid": "3ba66cec-0b8d-42a0-8b29-686f9fabe176", "_uuid": "3b95a98b8c11cfb076d6dd314b7808e7e30723ba"}
#Just the indirect features -- meta features
print("Using only Indirect features")
model = LogisticRegression(C=3)
X_train, X_valid, y_train, y_valid = train_test_split(target_x, target_y, test_size=0.33, random_state=2018)
train_loss = []
valid_loss = []
importance=[]
preds_train = np.zeros((X_train.shape[0], len(y_train)))
preds_valid = np.zeros((X_valid.shape[0], len(y_valid)))
for i, j in enumerate(TARGET_COLS):
    print('Class:= '+j)
    model.fit(X_train,y_train[j])
    preds_valid[:,i] = model.predict_proba(X_valid)[:,1]
    preds_train[:,i] = model.predict_proba(X_train)[:,1]
    train_loss_class=log_loss(y_train[j],preds_train[:,i])
    valid_loss_class=log_loss(y_valid[j],preds_valid[:,i])
    print('Trainloss=log loss:', train_loss_class)
    print('Validloss=log loss:', valid_loss_class)
    importance.append(model.coef_)
    train_loss.append(train_loss_class)
    valid_loss.append(valid_loss_class)
print('mean column-wise log loss:Train dataset', np.mean(train_loss))
print('mean column-wise log loss:Validation dataset', np.mean(valid_loss))

end_time=time.time()
print("total time till Indirect feat model",end_time-start_time)

# + {"_cell_guid": "a3ebe542-b900-4de0-be95-ec0e6d234d0a", "_uuid": "cabc31fc15323ccfe0b24cee79c8aacc37f1bbde"}
importance[0][0]

# + {"_cell_guid": "2f8b15a5-fa87-450d-a2ef-79d902b345dc", "_uuid": "20d1a5c857e0b718f561f4da7289ffd14a06a35c"}
plt.figure(figsize=(16,22))
plt.suptitle("Feature importance for indirect features",fontsize=20)
gridspec.GridSpec(3,2)
plt.subplots_adjust(hspace=0.4)
plt.subplot2grid((3,2),(0,0))
sns.barplot(SELECTED_COLS,importance[0][0],color=color[0])
plt.title("class : Toxic",fontsize=15)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)

plt.subplot2grid((3,2),(0,1))
sns.barplot(SELECTED_COLS,importance[1][0],color=color[1])
plt.title("class : Severe toxic",fontsize=15)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)

plt.subplot2grid((3,2),(1,0))
sns.barplot(SELECTED_COLS,importance[2][0],color=color[2])
plt.title("class : Obscene",fontsize=15)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)



plt.subplot2grid((3,2),(1,1))
sns.barplot(SELECTED_COLS,importance[3][0],color=color[3])
plt.title("class : Threat",fontsize=15)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)


plt.subplot2grid((3,2),(2,0))
sns.barplot(SELECTED_COLS,importance[4][0],color=color[4])
plt.title("class : Insult",fontsize=15)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)


plt.subplot2grid((3,2),(2,1))
sns.barplot(SELECTED_COLS,importance[5][0],color=color[5])
plt.title("class : Identity hate",fontsize=15)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)


# plt.subplot2grid((4,2),(3,0),colspan=2)
# sns.barplot(SELECTED_COLS,importance[6][0],color=color[0])
# plt.title("class : Clean",fontsize=15)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=90)
# plt.xlabel('Feature', fontsize=12)
# plt.ylabel('Importance', fontsize=12)

plt.show()

# + {"_cell_guid": "dc7dcb62-f94f-4434-b102-55e5154df008", "_uuid": "7701af32331d3f85f106a55e6edbe0725152fb96"}
from scipy.sparse import csr_matrix, hstack

#Using all direct features
print("Using all features except leaky ones")
target_x = hstack((train_bigrams,train_charngrams,train_unigrams,train_feats[SELECTED_COLS])).tocsr()


end_time=time.time()
print("total time till Sparse mat creation",end_time-start_time)
# -

target_x.toarray()

# + {"_cell_guid": "b073051d-b8bb-43eb-8382-5e4d4677d1d0", "_uuid": "abcdafffbcb9cdfa42011777a9107d9d594949b3"}
model = NbSvmClassifier(C=4, dual=True, n_jobs=-1)
X_train, X_valid, y_train, y_valid = train_test_split(target_x, target_y, test_size=0.33, random_state=2018)
train_loss = []
valid_loss = []
preds_train = np.zeros((X_train.shape[0], len(y_train)))
preds_valid = np.zeros((X_valid.shape[0], len(y_valid)))
for i, j in enumerate(TARGET_COLS):
    print('Class:= '+j)
    model.fit(X_train,y_train[j])
    preds_valid[:,i] = model.predict_proba(X_valid)[:,1]
    preds_train[:,i] = model.predict_proba(X_train)[:,1]
    train_loss_class=log_loss(y_train[j],preds_train[:,i])
    valid_loss_class=log_loss(y_valid[j],preds_valid[:,i])
    print('Trainloss=log loss:', train_loss_class)
    print('Validloss=log loss:', valid_loss_class)
    train_loss.append(train_loss_class)
    valid_loss.append(valid_loss_class)
print('mean column-wise log loss:Train dataset', np.mean(train_loss))
print('mean column-wise log loss:Validation dataset', np.mean(valid_loss))


end_time=time.time()
print("total time till NB base model creation",end_time-start_time)

# + {"_cell_guid": "f21bbb18-6193-4bbe-b2f4-72f0f16b67fc", "_uuid": "d3a38ead7ab890648b2e1338c401704e76f3eeea", "cell_type": "markdown"}
# ## Topic modeling:
# Due to kernal limitations(kernal timeout at 3600s), I had to continue the exploration in a seperate kernal( [Understanding the "Topic" of toxicity](https://www.kaggle.com/jagangupta/understanding-the-topic-of-toxicity)) to aviod a timeout. 
# # Next steps:
# * Add Glove vector features
# * Explore sentiement scores
# * Add LSTM, LGBM
#
# ## To be continued

# + {"_cell_guid": "ea348326-1575-4dc9-99b9-1df635777977", "_uuid": "5474c87d567d52dfda9a0acfc61e4b6d0e073575", "cell_type": "markdown"}
#
