#!/usr/bin/env python
# coding: utf-8

# ## In this project, we perform Text Generation using Language Modelling. We will see how to compute probabilities of N-grams and pick the next most probable word for generating text. We are using Shakespear Dataset for generating newer Text

# In[80]:


# import libraries
import numpy as np
import pandas as pd
import string
from nltk.tokenize import word_tokenize
import re
from nltk import ngrams
import random


# In[30]:


data=pd.read_csv("Data3/Shakespeare_data.csv")
data_array=[x for x in data.PlayerLine]


# In[31]:


data_array


# In[34]:


def preprocessing(data):
    
    clean="".join(v for v in data if v not in string.punctuation).lower()
    
    # Remove digits
    pattern='[0-9]'
    clean="".join(re.sub(pattern, ' ', i) for i in clean)
#     clean = clean.apply(lambda i: re.sub(pattern, ' ', i))
    return clean


# In[35]:


cleaned_data=[preprocessing(x) for x in data_array]


# In[37]:


cleaned_data


# In[42]:


appended_data=""
for i in cleaned_data:
    appended_data+=i
    appended_data+=" "


# In[43]:


appended_data


# Tokenizing Words

# In[44]:


Tokens=word_tokenize(appended_data) 
Tokens_nodupicate=np.unique(Tokens)
print(len(Tokens_nodupicate))


# In[66]:





# ### Creating N-Grams

# #### Unigrams

# In[47]:


# Creating Unigram Vocabulary
unigram_dict={}
[unigram_dict.update({x: 0}) for x in Tokens_nodupicate]


# In[67]:


for word in Tokens_nodupicate: #for each token in vocbulary
    unigram_dict[word]=appended_data.count(word)


# In[ ]:





# In[69]:


# finding probabilities of unigrams

N= sum(unigram_dict.values())# total no of words in corpus
probability={x:unigram_dict[x]/N for x in unigram_dict}


# In[70]:


print("Probability of unique unigrams")
probability


# #### Bigrams

# In[73]:


# Generate list of bigrams
bigram_tokens=np.array([])

n_grams = ngrams(appended_data.split(), 2)
bigram_tokens=np.append(bigram_tokens,[ ' '.join(grams) for grams in n_grams])
# Unique tokens
bigram_tokens=np.unique(bigram_tokens)

bigram_dict={}
[bigram_dict.update({x: 0}) for x in bigram_tokens]

for word in bigram_tokens: #for each token in vocbulary
    bigram_dict[word]=appended_data.count(word) 
        


# In[ ]:





# In[75]:


# Find probability
def bigram_probability(x):
#     finding P(ABC)=count(ABC)/count(AB)
    try:
        x=bigram_dict[x]/unigram_dict[x.rsplit(' ', 1)[0]]
    except:
        x=0 
    return x

probability_bigram={x:bigram_probability(x) for x in bigram_dict}

print("Probabilities for bigrams:")
probability_bigram


# #### Trigrams

# In[76]:


# Generate list of trigrams
trigrams_tokens=np.array([])
n_grams = ngrams(appended_data.split(), 3)
trigrams_tokens=np.append(trigrams_tokens,[ ' '.join(grams) for grams in n_grams])

#unique tokens
trigrams_tokens=np.unique(trigrams_tokens)

trigrams_dict={}
[trigrams_dict.update({x: 0}) for x in trigrams_tokens]

for word in trigrams_tokens: #for each token in vocbulary
    trigrams_dict[word]=appended_data.count(word) 


# In[77]:


# Find probability
def trigram_probability(x):
#     finding P(ABC)=count(ABC)/count(AB)
    try:
        x=trigrams_dict[x]/bigram_dict[x.rsplit(' ', 1)[0]]
    except:
        x=0 
    return x

probability_trigram={x:trigram_probability(x) for x in trigrams_dict}

print("Probabilities for trigrams:")
probability_trigram


# #### Text Generation

# Lets generate a few lines. The technique used below is backoff which means if if a required N-gram doesnt exist then back off to N-1 Gram and check if we have found N-1 gram word. Ultimately if no N-grams is found, we will always land to unigrams

# In[78]:


def generate_sentence(no_of_words):
    
#     First word is randomnly chosen from unigrams
    sentence=random.choice(list(unigram_dict.keys()))
#     Second word from bigrams by selecting the 1 bigram from 5 randomnly chosen highest probability keys

    prob={key:probability_bigram[key] for key in probability_bigram.keys() if key.rsplit(' ', 1)[0]==sentence}
    
    value_key_pairs = ((value, key) for (key,value) in  prob.items())
    sorted_value_key_pairs = sorted(value_key_pairs, reverse=True)
    sorted_dict={k: v for v, k in sorted_value_key_pairs}

    N = 5
    # Get first 5 keys-items in dictionary 
    d1= dict(list(sorted_dict.items())[0: N]) 
    
    sentence+=" "

#                     Randomnly choose any top probable key
    sentence+=(random.choice(list(d1.keys())).rsplit(' ', 1)[1])
    
    while len(sentence.split())<no_of_words-2: # as t2 words have already been added
        
        l=sentence.split(' ')
        x=l[len(l)-2]+" "+l[len(l)-1]
        try:
            prob={key:probability_trigram[key] for key in probability_trigram.keys() if key.rsplit(' ', 1)[0]==x}
           
            value_key_pairs = ((value, key) for (key,value) in  prob.items())
            sorted_value_key_pairs = sorted(value_key_pairs, reverse=True)
            sorted_dict={k: v for v, k in sorted_value_key_pairs}

            N = 5
            # Get first 5 keys-items in dictionary 
            d1= dict(list(sorted_dict.items())[0: N]) 

            sentence+=" "

        #                     Randomnly choose any top probable key
            sentence+=(random.choice(list(d1.keys())).rsplit(' ', 1)[1])
    
        except KeyError: # if KeyError occurs that means no key exist in trigram, so se back off technique to bigrams
            
            try:
                
                x2= l[len(l)-1]
                prob={key:probability_bigram[key] for key in probability_bigram.keys() if key.rsplit(' ', 1)[0]==x2}
                
                value_key_pairs = ((value, key) for (key,value) in  prob.items())
                sorted_value_key_pairs = sorted(value_key_pairs, reverse=True)
                sorted_dict={k: v for v, k in sorted_value_key_pairs}

                N = 5
                # Get first 5 keys-items in dictionary 
                d1= dict(list(sorted_dict.items())[0: N]) 
                
                sentence+=" "

            #                     Randomnly choose any top probable key
                sentence+=(random.choice(list(d1.keys())).rsplit(' ', 1)[1])
                
            except KeyError:  # if KeyError occurs that means no key exist in bigram, so se back off technique to uigrams
                        
                value_key_pairs = ((value, key) for (key,value) in unigram_dict.items())
                sorted_value_key_pairs = sorted(value_key_pairs, reverse=True)
                sorted_dict={k: v for v, k in sorted_value_key_pairs}

                N = 5
                # Get first 5 keys-items in dictionary 
                d1= dict(list(sorted_dict.items())[0: N]) 
                
                sentence+=" "

#                     Randomnly choose any top probable key
                sentence+=random.choice(list(d1.keys()))

                    
    return sentence


# In[93]:


# Generate New headlines
for i in range(0,5):
    no_of_words=random.randint(10, 20)
    generated_line= generate_sentence(no_of_words)
    print(i+1, ".",generated_line)



# In[ ]:





# In[ ]:




