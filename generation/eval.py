from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
import torch
from tqdm import tqdm
import re
import pandas as pd
from detoxify import Detoxify
import torch 
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import BertForTokenClassification, BertForSequenceClassification,BertPreTrainedModel, BertModel
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
import syllables
from collections import Counter
from empath import Empath

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time'],
    # terms that will be annotated
    fix_html=True,  # fix HTML tokens
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    #corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons])


def preprocess_func(text):
    remove_words=['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<elongated>','<emphasis>','<repeated>','\'','s']
    word_list=text_processor.pre_process_doc(text)
    word_list=list(filter(lambda a: a not in remove_words, word_list)) 
    sent=" ".join(word_list)
    sent = re.sub(r"[<\*>]", " ",sent)
    word_list=sent.split(" ")
    return word_list



def hate_refrences(data,test_set):          ###############returns pair of <hate,refrences>  
    hate  = []
    reply = []
    refrences = []
    for sample in data:
        ht , rep = sample[0] , sample[1]
        hate.append(ht)
        reply.append(rep)
    hate = list(set(hate))
    mp={}
    for ht_i in hate:
        refs = []
        for sample in data:
            ht_j , rep =  sample[0] , sample[1]
            if ht_j == ht_i:
                refs.append(rep)
        mp[ht_i] = refs
        refrences.append(refs)
    hate = list(set([x[0] for x in test_set]))
    refs = [mp[ht_i] for ht_i in hate]
    return hate,refs             # a given hate instance and refrences(replies) for metrics evaluation


# In[7]:


def training_corpus(train_set):    # returns training corpus
    replies = []
    for sample in train_set:
        rep = sample[1]
        replies.append(rep)
    replies = list(set(replies))
    return replies                # returns the sentences used while training 


def tokenize(sentence, tokenizer, max_sequence_length=None):
    token_sent = list(map(lambda x: str(x), list(tokenizer.tokenize(sentence))))
    if max_sequence_length is None:
        return token_sent
    else:
        return token_sent[:max_sequence_length]


def evaluate(params, model, test_dataloader, tokenizer, device):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Evaluating"):
        inputs, labels = (batch[0], batch[0])
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels[labels == tokenizer.pad_token_id] = -100
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1
        
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    return perplexity




def dummy(list_sent):
    return list_sent

###################################### BLEU_SCORE , METEOR #######################################
from nltk import word_tokenize

def nltk_metrics(params,tokenizer):
    hypo = params[0]  # a list of generated_hypothesis   
    refs = params[1]  # a list of refrences for particular_refrences    
    
    hypo = [tokenize(h, tokenizer) for h in hypo] 
    refs = [[tokenize(r, tokenizer) for r in ref] for ref in refs]
    
    
#     hypo = [word_tokenize(h) for h in hypo] 
#     refs = [[word_tokenize(r) for r in ref] for ref in refs]
   
    
    
    gleu= bleu = meteor_ = 0.0
    
    print(len(hypo))
    for step in tqdm(range(len(hypo)),desc='Nltk metrics cal:'):
        ref = refs[step]
        hyp = hypo[step]
        
#         print("References",ref[0])
#         print("Hypothesis", hyp)
#         print("===========================")
        bleu  += sentence_bleu(ref,hyp, weights=(1.0,1.0,0,0,0.0), smoothing_function=SmoothingFunction().method4)
        gleu += sentence_gleu(ref,hyp,min_len=1,max_len=2)
        meteor_ += meteor_score(params[1][step], params[0][step])
#         meteor_ += meteor_score(params[1][step], params[0][step])
    if len(hypo) == 0:
        return 0
    gleu  /= len(hypo)
    bleu  /= len(hypo)
    meteor_ /= len(hypo)
    
    return bleu,gleu,meteor_



############################################ JACCARD SIMILARITY #################################
def get_jaccard_sim(str1, str2):   
    if isinstance(str1, float) or isinstance(str2, float):
        return (-1)
    try:
        a = set(str1.split()) 
        b = set(str2.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
    except:
        print((str1))
        print(type(str2))
        return 0


############################################### NOVELTY #########################################
def get_novelty(sent,training_corpus):
    max_overlap = 0
    for instance in training_corpus:
        max_overlap = max(max_overlap,get_jaccard_sim(instance,sent))
    return 1-max_overlap

def avg_novelty(sentences,training_corpus):
    avg = 0
    for sent in tqdm(sentences,total=len(sentences),desc='Novelty:'):
        avg += get_novelty(sent,training_corpus)
    avg = (avg/float(len(sentences)))
    return avg



############################################### DIVERSITY ########################################
def get_diversity(sentences):
    avg = 0.0
    for i in tqdm(range(len(sentences)),desc='Diversity:'):
        max_overlap = 0
        for j in range(len(sentences)):
            if i!=j:
                max_overlap = max(max_overlap,get_jaccard_sim(sentences[i],sentences[j]))
        avg = avg + (1-max_overlap)
    avg = (avg/len(sentences))
    return avg
    
def diversity_and_novelty(training_corpus,gen_replies):
    diversity = get_diversity(gen_replies)
    novelty   = avg_novelty(gen_replies,training_corpus)
    return diversity,novelty





############################################## HEAVY METRICS ########################################
class Bleurt():
    def __init__(self, model_path,cache_path,max_length, batch_size,use_gpu):
        self.max_length= max_length
        self.batch_size = batch_size
        self.use_gpu=use_gpu
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,cache_dir=cache_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,cache_dir=cache_path)
        self.device = torch.device("cpu")
        if(self.use_gpu):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        self.model.eval()
    
   
    def score(self,params):
        hypo = params[0]  # a list of generated_hypothesis   
        refs = params[1]  # a list of refrences for particular_refrences    
        device = self.device
        list_ids=[]
        hypo_all=[]
        refs_all=[]
        for step in range(len(hypo)):
            for ref in refs[step]:
                
                list_ids.append(step)
                hypo_all.append(hypo[step])
                refs_all.append(ref)
        
        print("Collected all points")
        scores_all=[]
        for i in tqdm(range(0, len(hypo_all), self.batch_size)):
            with torch.no_grad():
              if(len(refs_all[i:i+self.batch_size])==1):
                    continue
                  
            
            
            
              inputs = self.tokenizer(refs_all[i:i+self.batch_size], hypo_all[i:i+self.batch_size], return_tensors='pt',truncation=True, padding=True, max_length=self.max_length)
              
              
                
              if(self.use_gpu):   
                    scores = list(self.model(input_ids=inputs['input_ids'].to(device),
                                           attention_mask=inputs['attention_mask'].to(device),
                                           token_type_ids=inputs['token_type_ids'].to(device))[0].squeeze().cpu().numpy())
              else:
                  scores = list(self.model(input_ids=inputs['input_ids'],
                                           attention_mask=inputs['attention_mask'],
                                           token_type_ids=inputs['token_type_ids'])[0].squeeze().cpu().numpy())
              
            
              scores_all+=scores
        
        
        
        df=pd.DataFrame(list(zip(list_ids, scores_all)), columns=['ids', 'scores'])
        df_mean=df.groupby(['ids']).mean()        
        
        print(df_mean.head(5))
        
        mean_bleurt_score = np.mean(list(df_mean['scores']))
        return mean_bleurt_score

    
#### Without REFERENCE
class Argument_scoring():
    def __init__(self, model_path,cache_path,max_length, batch_size,use_gpu):
        self.max_length= max_length
        self.batch_size = batch_size
        self.use_gpu=use_gpu
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,cache_dir=cache_path,use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,cache_dir=cache_path)
        self.device=torch.device("cpu")
        if(self.use_gpu):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        self.model.eval()
    
    def scoring(self, hypo):
        device = self.device
        scores_all=[]
        for i in tqdm(range(0, len(hypo), self.batch_size)):
            with torch.no_grad():
            
              inputs = self.tokenizer(hypo[i:i+self.batch_size],return_tensors='pt',truncation=True, padding=True, max_length=self.max_length)
              
              if(self.use_gpu):    
                  scores = self.model(input_ids=inputs['input_ids'].to(device),attention_mask=inputs['attention_mask'].to(device))[0].squeeze()
              else:
                  scores = self.model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])[0].squeeze()
              scores = torch.softmax(scores.T, dim=0).T.cpu().numpy()
              scores_all+=list(scores[:,1])
        
        
#         with torch.no_grad():
#             scores = self.model(**self.tokenizer(hypo, return_tensors='pt',truncation=True, padding=True, max_length=64))[0].squeeze()
#             scores = torch.softmax(scores.T, dim=0).T.cpu().numpy()
        
        return np.mean(scores_all)
    

    

class Dialog_upvote_scoring():
    def __init__(self, model_path,cache_path,max_length, batch_size,use_gpu):
        self.max_length= max_length
        self.batch_size = batch_size
        self.use_gpu=use_gpu
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,cache_dir=cache_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,cache_dir=cache_path)
        self.device=torch.device("cpu")
        if(self.use_gpu):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        self.model.eval()
    
    def scoring(self,hypo,hate):
        device = self.device
        hypo_hate=[]
        for i in range(len(hypo)):
            str1=hate[i]+'<|endoftext|>'+hypo[i]
            hypo_hate.append(str1)
        
        device = self.device

        scores_all=[]
        for i in tqdm(range(0, len(hypo_hate), self.batch_size)):
            with torch.no_grad():
                inputs = self.tokenizer(hypo_hate[i:i+self.batch_size],return_tensors='pt',truncation=True, padding=True, max_length=self.max_length)
                if(self.use_gpu):    
                      results = self.model(input_ids=inputs['input_ids'].to(device),attention_mask=inputs['attention_mask'].to(device),return_dict=True)
                else:
                      results = self.model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],return_dict=True)
                scores=list(torch.sigmoid(results.logits).cpu().numpy())
                scores_all+=scores
        print(scores[0:5])
        return np.mean(scores_all)


class Counter_argument_scoring():
    def __init__(self, model_path,cache_path,max_length, batch_size,use_gpu):
        self.max_length= max_length
        self.batch_size = batch_size
        self.use_gpu=use_gpu
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,cache_dir=cache_path,use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,cache_dir=cache_path)
        self.device = torch.device("cpu")
        if(self.use_gpu):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        self.model.eval()
    
    def scoring(self,hypo,hate):
        device = self.device
        scores_all=[]
        for i in tqdm(range(0, len(hypo), self.batch_size)):
            with torch.no_grad():
                inputs = self.tokenizer(text=hate[i:i+self.batch_size],text_pair=hypo[i:i+self.batch_size],return_tensors='pt',truncation=True, padding=True, max_length=self.max_length)
                if(self.use_gpu):    
                      scores = self.model(input_ids=inputs['input_ids'].to(device),attention_mask=inputs['attention_mask'].to(device),return_dict=True)
                else:
                      scores = self.model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],return_dict=True)
                scores = torch.softmax(scores['logits'].T, dim=0).T.cpu().numpy()
                scores_all+=list(scores[:,1])
        
        print(scores_all[0:5])
        return np.mean(scores_all)


class Toxicity_model():
    def __init__(self,max_length, batch_size,use_gpu):
        self.max_length= max_length
        self.batch_size = batch_size
        self.use_gpu=use_gpu
        self.device = torch.device("cpu")
        if(self.use_gpu):
            self.device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
            self.model = Detoxify('unbiased', device='cuda:0')
        else:
            self.model = Detoxify('unbiased', device='cpu')
        
    def scoring(self,hypo):
        scores_all=[]
        for i in tqdm(range(0, len(hypo), self.batch_size)):
            with torch.no_grad():
                scores=self.model.predict(hypo[i:i+self.batch_size])
                scores_all+=list(scores['toxicity'])
        print(scores_all[0:5])
        return np.mean(scores_all)
    
    
###################################### Detox scores using HateXplain #################################################
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
class Model_Rational_Label(BertPreTrainedModel):
     def __init__(self,config):
        super().__init__(config)
        self.num_labels=2
        self.impact_factor=0.8
        self.bert = BertModel(config,add_pooling_layer=False)
        self.bert_pooler=BertPooler(config)
        self.token_dropout = nn.Dropout(0.1)
        self.token_classifier = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()        
#         self.embeddings = AutoModelForTokenClassification.from_pretrained(params['model_path'], cache_dir=params['cache_path'])
        
     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, attn=None, labels=None):
        outputs = self.bert(input_ids, attention_mask)
        # out = outputs.last_hidden_state
        out=outputs[0]
        logits = self.token_classifier(self.token_dropout(out))
        
        
#         mean_pooling = torch.mean(out, 1)
#         max_pooling, _ = torch.max(out, 1)
#         embed = torch.cat((mean_pooling, max_pooling), 1)
        embed=self.bert_pooler(outputs[0])
        y_pred = self.classifier(self.dropout(embed))
        loss_token = None
        loss_label = None
        loss_total = None
        
        if attn is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if mask is not None:
                active_loss = mask.view(-1) == 1
                active_logits = logits.view(-1, 2)
                active_labels = torch.where(
                    active_loss, attn.view(-1), torch.tensor(loss_fct.ignore_index).type_as(attn)
                )
                loss_token = loss_fct(active_logits, active_labels)
            else:
                loss_token = loss_fct(logits.view(-1, 2), attn.view(-1))
            
            loss_total=self.impact_factor*loss_token
            
            
        if labels is not None:
            loss_funct = nn.CrossEntropyLoss()
            loss_logits =  loss_funct(y_pred.view(-1, self.num_labels), labels.view(-1))
            loss_label= loss_logits
            if(loss_total is not None):
                loss_total+=loss_label
            else:
                loss_total=loss_label
        if(loss_total is not None):
            return y_pred, logits, loss_total
        else:
            return y_pred, logits

# class Args():
#     def __init__(self):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# args = Args()

class Toxic_HateXplain_scoring():
    def __init__(self, model_path,cache_path,max_length, batch_size,use_gpu):
        self.max_length= max_length
        self.batch_size= batch_size
        self.use_gpu   = use_gpu
        self.tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two",cache_dir=cache_path)
        self.model     =  Model_Rational_Label.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two")
        self.device = torch.device("cpu")
        if(self.use_gpu):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device) 
        self.model.eval()
    
    def scoring(self,hypo,hate):
        scores_all=[]
        device = self.device
        for i in tqdm(range(0, len(hypo), self.batch_size)):
            with torch.no_grad():
                inputs = self.tokenizer(text=hypo[i:i+self.batch_size],return_tensors='pt',truncation=True, padding=True, max_length=self.max_length)
                if(self.use_gpu):    
                      logits, _ = self.model(input_ids=inputs['input_ids'].to(device),attention_mask=inputs['attention_mask'].to(device))
                else:
                      logits, _ = self.model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])
                scores = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
                scores_all+=list(scores[:,1])
        
        print(scores_all[0:5],len(scores_all))
        return np.mean(scores_all)
        
###################################### Flesch Reading Ease #################################################

def fre(para):
    '''Flesch Reading Ease
    Arguments:
        nsyll:  syllable count
        nwords:  word count
        nsentences:  sentence count
    Returns:
        float:  Flesch reading ease score
    '''
    nsentences = len(sent_tokenize(para))
    words = para.split()
    nwords = len(words)
    nsyll = 0
    for word in words:
        nsyll += syllables.estimate(word)
    try:
        return 206.835 - (84.6 * (nsyll / nwords)) - (1.015 * (nwords / nsentences))
    except ZeroDivisionError:
        return 0
    

def fre_readability(sentences):
    avg = 0.0
    for i in tqdm(range(len(sentences)),desc='Flesch Reading Ease:'):
        avg +=fre(sentences[i])
    if len(sentences) == 0:
        return 0
    avg = (avg/len(sentences))
    return avg

###################################### Empath #################################################
def empath(sentences):
    lexicon = Empath()
    final_dict = Counter()
    for i in tqdm(range(len(sentences)),desc='Empath:'):
        emp =  lexicon.analyze(sentences[i], categories=['help', 'hate', 'cheerfulness', 'aggression', 'dispute', 'optimism', 'healing', 'love', 'sympathy', 'politeness', 'fight', 'disgust', 'warmth', 'valuable', 'sadness', 'emotional', 'joy', 'anger', 'shame', 'affection', 'negotiate', 'positive_emotion'], normalize=True)
        for key, value in emp.items():
            final_dict[key] += value

    final_dict = {k:v/len(sentences) for k,v in dict(final_dict).items()}
    return final_dict

###################################### Stylistic Element ######################################
def avg_words(sentences):
    avg_1=0
    avg_2=0
    for sentence in sentences:
        num_of_words=len(sentence.split())
        avg_1=avg_1+ num_of_words
        avg_2+= (num_of_words)*(num_of_words)
    avg_1/= len(sentences)
    avg_2/= len(sentences)
    var= avg_2-(avg_1*avg_1)
    mean_var={
              "word_mean":avg_1,
              "word_variance": var
             }
    return mean_var

def avg_punctuation(sentences):
    avg_1=0
    avg_2=0
    for sentence in sentences:
        num=0
        for c in sentence:
            if (c==':'):
                num+=1;
            if(c==';'):
                num+=1
            if(c==','):
                num+=1
        avg_1+=num
        avg_2+=num*num
    avg_1/=len(sentences)
    avg_2/= len(sentences)
    var= avg_2-avg_1*avg_1
    mean_var={
              "punc_mean":avg_1,
              "punc_variance": var
             }
    return mean_var



