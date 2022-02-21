import json
import pandas as pd 
from gensim.parsing.preprocessing import remove_stopwords
import preprocessor as p
from gensim.utils import tokenize

source = pd.read_json('./dataset/source.json')
reaction = pd.read_json('./dataset/reaction.json')

with open('./dataset/annotation.json', 'r') as f:
    annotation = json.load(f)
annotation = pd.DataFrame(annotation)

source = source.rename({'id':'reply_id','text':'text_source'},axis='columns')
source = pd.concat([source, annotation['is_rumour']],axis=1)

reaction['in_reply_to_status_id_str'] = reaction['in_reply_to_status_id_str'].fillna(0)
reaction = reaction.astype({'in_reply_to_status_id_str':'int64'})
reaction = reaction.rename({'in_reply_to_status_id_str':'reply_id'},axis='columns')

df = pd.merge(reaction, source, on='reply_id', how='left')
df = df.drop(df[df['text_source'].isna() == True].index,axis=0)
df = df.reset_index(drop=True)

df[['text','text_source','is_rumour']].to_json('./dataset/tweets.json', orient='records')

# trimming tweets

df = pd.read_json('./dataset/tweets.json')

def preprocess(x):
    x = p.clean(x)
    x = remove_stopwords(x.lower())
    x = list(tokenize(x))
    return x

df['text'] = df['text'].map(lambda x: preprocess(x))
df['text_source'] = df['text_source'].map(lambda x: preprocess(x))

df = df.loc[df['text'].str.len() > 3]
df = df.reset_index(drop=True)

df['label'] = df['is_rumour'].replace({'rumour': 1, 'nonrumour': 0})

df.to_json('./dataset/tweets.json', orient='records')