from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import re
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
import math
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




class Normal_Generation_Dataset():
    def __init__(self, data, tokenizer=None,  params=None,train = False):
        self.data = data
        self.params= params
        self.batch_size = self.params['batch_size']
        self.train = train
        self.max_length=params['max_length']        
        self.count=0
        self.count_dic = {}
        self.tokenizer = tokenizer
        self.inputs = self.process_data(self.data)
        self.DataLoader = self.get_dataloader(self.inputs)
        
    def preprocess_func(self, text):
        remove_words=['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<elongated>','<emphasis>','<repeated>','\'','s']
        word_list=text_processor.pre_process_doc(text)
        word_list=list(filter(lambda a: a not in remove_words, word_list)) 
        sent=" ".join(word_list)
        sent = re.sub(r"[<\*>]", " ",sent)
        sent = text
        
        return sent
    
    def convert_stance(self,stance):
        if(stance=='Pro'):
            return self.params['pro_token']
        elif(stance=='Against'):
            return self.params['con_token']
        else:
            return self.params['neutral_token']
    
    
    def convert_topic(self,label):
        if(label=='Hate'):
            return '<|HATE|>'
        elif(label=='Counterspeech'):
            return '<|COUN|>'
        else:
            return self.params['neutral_token']

    def add_template(self,initiator,row):
        
        init_stance='initiator_stance_converted'
        reply_stance='reply_stance_converted'
        ml=self.max_length-150
        
#         print(initiator)
        tokenized=self.tokenizer.encode(initiator,truncation=True,max_length=ml)
        hate_sentence=self.tokenizer.decode(tokenized,skip_special_tokens=True)
        
        if(self.params['template_version']=='token'):
            if(self.params['stance']==True):
                stance_template = self.convert_stance(row[init_stance]) + " " + self.convert_stance(row[reply_stance])
            else:
                stance_template = ""
            
            if(self.params['topic']==True):
                topic_template = " <|TOS|> "+row['title'][:-1]+" <|TOE|> "
            else:
                topic_template = ""
            
            
            if(self.params['label']==True):
                label_template = self.convert_topic(row['label'])
            else:
                label_template = ""
            
            hate_sentence = hate_sentence + topic_template + label_template + stance_template
           
        if(self.params['template_version']=='sentence'):
            if(self.params['stance']==True):
                hate_sentence = " and " + self.convert_stance(row[init_stance]) + " " + hate_sentence +" then " + self.convert_stance(row[reply_stance])
            if(self.params['topic']==True):
                if(self.params['label']==True):
                    hate_sentence = self.params['topic_head']+"-"+row['title'][:-1]+" is "+row['label']+" "+hate_sentence
                else:
                    hate_sentence = self.params['topic_head']+"-"+row['title'][:-1]+" "+hate_sentence
        return hate_sentence
    
    
    
    
    def add_template_finetuning(self,initiator,row):
        ml=self.max_length-150
        tokenized=self.tokenizer.encode(initiator,truncation=True,max_length=ml)
        hate_sentence=self.tokenizer.decode(tokenized,skip_special_tokens=True)
        if(self.params['template_version']=='token'):
            if(self.params['stance']==True):
                stance_template = self.params['pro_token']  + " " + self.params['con_token'] 
            else:
                stance_template = ""
            
            if(self.params['topic']==True):
                topic_template = " <|TOS|> "+row[self.params['type_of_data']]+" <|TOE|> "
            else:
                topic_template = ""
            
            
            if(self.params['label']==True):
                label_template = self.convert_topic("Hate")
            else:
                label_template = ""
            
            hate_sentence = hate_sentence + topic_template + label_template + stance_template
           
        if(self.params['template_version']=='sentence'):
            if(self.params['stance']==True):
                hate_sentence = " and " + self.params['pro_token'] + " " + hate_sentence +" then " + self.params['con_token']
            if(self.params['topic']==True):
                if(self.params['label']==True):
                    hate_sentence = self.params['topic_head']+"-"+row[self.params['type_of_data']]+" is "+"Hate"+" "+hate_sentence
                else:
                    hate_sentence = self.params['topic_head']+" is "+row[self.params['type_of_data']]+hate_sentence
        
        return hate_sentence

    
    def construct_conv(self,dict_reply_pair):
        
        conv = None
        flatten = lambda l: [item for sublist in l for item in sublist]
        initiator=self.preprocess_func(dict_reply_pair['initiator_message'])
        reply=self.preprocess_func(dict_reply_pair['reply_message'])
        
        
        if(self.params['type']=='finetuning'):
            initiator =  self.add_template_finetuning(initiator,dict_reply_pair)
        elif(self.params['type']=='pretraining'):
            initiator =  self.add_template(initiator,dict_reply_pair)
        
        
        
        conv_initiator=self.tokenizer.encode(initiator)
        
        reply_length=int((self.max_length)-1-len(conv_initiator))
        
        #print("Initiator:",len(conv_initiator),"Reply:",reply_length)
        
        if(reply_length <= 0):
            self.count+=1
            conv_initiator=self.tokenizer.encode(initiator)+[self.tokenizer.eos_token_id]
            reply_length=int((self.max_length)-1-len(conv_initiator))
            
        reply_argument = self.tokenizer.encode(reply,truncation=True,max_length=reply_length)+[self.tokenizer.eos_token_id]
        
        conv = conv_initiator + reply_argument
#         print(" ".join(x for x in self.tokenizer.decode(conv)))
#         print(100*'#')
        return conv

    def tokenize(self, dataframe):
        inputs=[]
        print(self.count)
        import pandas as pd
        for index,row in tqdm(dataframe.iterrows(),total=len(dataframe)):
            if pd.isna(row['initiator_message']):
                continue
            conv = self.construct_conv(row)
            inputs.append(conv)
            
        print(self.count)
#         print("%"*100)
        return inputs
    
    def process_data(self, data):
        inputs = self.tokenize(data)
        return inputs
    
    def get_dataloader(self, inputs):
        inputs = pad_sequences(inputs,maxlen=int(self.params['max_length']), dtype="long", 
                          value=self.tokenizer.pad_token_id, truncating="post", padding="pre")
        inputs = torch.tensor(inputs)
        data = TensorDataset(inputs)
        if self.train:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size, drop_last=True)

    
    
    
    
    
    
class Normal_T5_Generation_Dataset():
    def __init__(self, data, tokenizer=None,  params=None,train = False):
        self.data = data
        self.params= params
        self.batch_size = self.params['batch_size']
        self.train = train
        self.max_length=params['max_length']        
        self.count=0
        self.count_dic = {}
        self.tokenizer = tokenizer
        self.inputs = self.process_data(self.data)
        self.DataLoader = self.get_dataloader(self.inputs)
        
    def preprocess_func(self, text):
        remove_words=['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<elongated>','<emphasis>','<repeated>','\'','s']
        word_list=text_processor.pre_process_doc(text)
        word_list=list(filter(lambda a: a not in remove_words, word_list)) 
        sent=" ".join(word_list)
        sent = re.sub(r"[<\*>]", " ",sent)
        return sent
    
    def convert_stance(self,stance):
        if(stance=='Pro'):
            return self.params['pro_token']
        elif(stance=='Against'):
            return self.params['con_token']
        else:
            return self.params['neutral_token']
    
    
    def convert_topic(self,label):
        if(label=='Hate'):
            return '<|HATE|>'
        elif(label=='Counterspeech'):
            return '<|COUN|>'
        else:
            return self.params['neutral_token']

    def add_template(self,initiator,row):
        
        init_stance='initiator_stance_converted'
        reply_stance='reply_stance_converted'
        ml=self.max_length-150
        tokenized=self.tokenizer.encode(initiator,truncation=True,max_length=ml)
        hate_sentence=self.tokenizer.decode(tokenized,skip_special_tokens=True)
        
        if(self.params['template_version']=='token'):
            if(self.params['stance']==True):
                stance_template = self.convert_stance(row[init_stance]) + " " + self.convert_stance(row[reply_stance])
            else:
                stance_template = ""
            
            if(self.params['topic']==True):
                topic_template = " <|TOS|> "+row['title'][:-1]+" <|TOE|> "
            else:
                topic_template = ""
            
            
            if(self.params['label']==True):
                label_template = self.convert_topic(row['label'])
            else:
                label_template = ""
            
            hate_sentence = hate_sentence + topic_template + label_template + stance_template
           
        if(self.params['template_version']=='sentence'):
            if(self.params['stance']==True):
                hate_sentence = " and " + self.convert_stance(row[init_stance]) + " " + hate_sentence +" then " + self.convert_stance(row[reply_stance])
            if(self.params['topic']==True):
                if(self.params['label']==True):
                    hate_sentence = self.params['topic_head']+"-"+row['title'][:-1]+" is "+row['label']+" "+hate_sentence
                else:
                    hate_sentence = self.params['topic_head']+"-"+row['title'][:-1]+" "+hate_sentence
        return hate_sentence
    
    
    
    
    def add_template_finetuning(self,initiator,row):
        ml=self.max_length-150
        tokenized=self.tokenizer.encode(initiator,truncation=True,max_length=ml)
        hate_sentence=self.tokenizer.decode(tokenized,skip_special_tokens=True)
        if(self.params['template_version']=='token'):
            if(self.params['stance']==True):
                stance_template = self.params['pro_token']  + " " + self.params['con_token'] 
            else:
                stance_template = ""
            
            if(self.params['topic']==True):
                topic_template = " <|TOS|> "+row[self.params['type_of_data']]+" <|TOE|> "
            else:
                topic_template = ""
            
            
            if(self.params['label']==True):
                label_template = self.convert_topic("Hate")
            else:
                label_template = ""
            
            hate_sentence = hate_sentence + topic_template + label_template + stance_template
           
        if(self.params['template_version']=='sentence'):
            if(self.params['stance']==True):
                hate_sentence = " and " + self.params['pro_token'] + " " + hate_sentence +" then " + self.params['con_token']
            if(self.params['topic']==True):
                if(self.params['label']==True):
                    hate_sentence = self.params['topic_head']+"-"+row[self.params['type_of_data']]+" is "+"Hate"+" "+hate_sentence
                else:
                    hate_sentence = self.params['topic_head']+" is "+row[self.params['type_of_data']]+hate_sentence
        
        return hate_sentence

    
    def construct_conv(self,dict_reply_pair):
        conv = None
        flatten = lambda l: [item for sublist in l for item in sublist]
        initiator=self.preprocess_func(dict_reply_pair['initiator_message'])
        reply=self.preprocess_func(dict_reply_pair['reply_message'])
        
        if(self.params['type']=='finetuning'):
            initiator =  self.add_template_finetuning(initiator,dict_reply_pair)
        elif(self.params['type']=='pretraining'):
            initiator =  self.add_template(initiator,dict_reply_pair)
        
        
        source = self.tokenizer.batch_encode_plus([initiator], max_length= 256, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
        
        reply_length=int((self.max_length)-1-len(source['input_ids'].squeeze()))
            
        
        target = self.tokenizer.batch_encode_plus([reply], max_length= reply_length, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
        
        
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        
        
        return {
        'source_ids': source_ids.to(dtype=torch.long), 
        'source_mask': source_mask.to(dtype=torch.long), 
        'target_ids': target_ids.to(dtype=torch.long),
        'target_ids_y': target_ids.to(dtype=torch.long)
        }


    def tokenize(self, dataframe):
        inputs=[]
        print(self.count)

        for index,row in tqdm(dataframe.iterrows(),total=len(dataframe)):
            conv = self.construct_conv(row)
            inputs.append(conv)
            
        print(self.count)
        return inputs
    
    def process_data(self, data):
        inputs = self.tokenize(data)
        return inputs
    
    def get_dataloader(self, inputs):
        inputs = pad_sequences(inputs,maxlen=int(self.params['max_length']), dtype="long", 
                          value=self.tokenizer.pad_token_id, truncating="post", padding="pre")
        inputs = torch.tensor(inputs)
        data = TensorDataset(inputs)
        if self.train:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size, drop_last=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

class Implication_Generation_Dataset(Normal_Generation_Dataset):
    
    def construct_conv(self,dict_reply_pair):
        conv = None
        flatten = lambda l: [item for sublist in l for item in sublist]
        initiator=self.preprocess_func(dict_reply_pair['hatespeech'])
        reply=self.preprocess_func(dict_reply_pair['implication'])
        reply_tokenized=[self.tokenizer.eos_token_id] + self.tokenizer.encode(reply)+[self.tokenizer.eos_token_id]
        conv = self.tokenizer.encode(initiator,truncation=True,max_length=int(self.max_length-len(reply_tokenized)))+reply_tokenized
        return conv
    
    
    
    
    
class Normal_Dexpert_Dataset(Normal_Generation_Dataset):
    def __init__(self, data, tokenizer=None,  params=None,train = False):
        self.params= params
        self.label = params['label']
        data=data[data['labels']==self.label]
        self.data = data
        self.batch_size = self.params['batch_size']
        self.train = train
        self.max_length=params['max_length']        
        self.count_dic = {}
        self.tokenizer = tokenizer
        self.inputs = self.process_data(self.data)
        self.DataLoader = self.get_dataloader(self.inputs)
    
    
    
    def construct_conv(self,dict_reply_pair):
        conv = None
        initiator=self.preprocess_func(dict_reply_pair['text'])
        conv = list(self.tokenizer.encode(initiator,truncation=True,max_length=int(self.max_length)))
        return conv

    
class Normal_Dexpert_Dataset_new(Normal_Generation_Dataset):
    def __init__(self, data, tokenizer=None,  params=None,train = False):
        self.params= params
        self.label = params['label']
        if(params['take_label']=='true'):
            data=data[data['labels']==self.label]
        else:
            data=data[data['labels']!=self.label]
        self.data = data
        self.batch_size = self.params['batch_size']
        self.train = train
        self.max_length=params['max_length']        
        self.count_dic = {}
        self.tokenizer = tokenizer
        self.inputs = self.process_data(self.data)
        self.DataLoader = self.get_dataloader(self.inputs)
    
    def tokenize(self, dataframe):
        inputs=[]
        past_left=[]
        for index,row in tqdm(dataframe.iterrows(),total=len(dataframe)):
            conv, past_left = self.construct_conv(row,past_left)
            inputs.append(conv)
        return inputs
    
    def construct_conv(self,dict_reply_pair,past_left):
        initiator=self.preprocess_func(dict_reply_pair['text'])
        conv = list(self.tokenizer.tokenize(initiator))
        conv = past_left+conv
        conv_new=self.tokenizer.convert_tokens_to_ids(conv[:self.max_length])
        past_left=conv[self.max_length:]
                    
        return conv_new,past_left