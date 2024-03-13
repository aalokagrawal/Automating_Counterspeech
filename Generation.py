import json
from os import listdir
import glob
from tqdm import tqdm
import transformers 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import pandas as pd
import numpy as np
import GPUtil
import time
import itertools
import argparse
from generation.data import *
debug=False
import random
from transformers import AutoTokenizer,AutoModelForCausalLM, AutoModelForSeq2SeqLM,BitsAndBytesConfig
# from Generation.model_gpt_j import *


def preprocess_func(text):
    remove_words=['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<elongated>','<emphasis>','<repeated>','\'','s']
    word_list=text_processor.pre_process_doc(text)
    word_list=list(filter(lambda a: a not in remove_words, word_list)) 
    sent=" ".join(word_list)
    sent = re.sub(r"[<\*>]", " ",sent)
    return sent



#task_name  is a list having element in the format (task_name,class_name)
def get_gpu(model_path):
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    if model_path == 'EleutherAI/gpt-j-6B':
        max_mem = 0.1
#         print("true")
    else:
        max_mem = 0.3
      
    while(1):
        tempID = [] 
        tempID = GPUtil.getAvailable(order = 'memory', limit = 2, maxLoad = 1.0, maxMemory = max_mem, includeNan=False, excludeID=[], excludeUUID=[])

        for i in range(len(tempID)):
            if len(tempID) > 0:
                print("Found a gpu")
                print('We will use the GPU:',tempID[i],torch.cuda.get_device_name(tempID[i]))
                deviceID=[tempID[i]]
                return deviceID
            else:
                time.sleep(5)

#template definitions
# def get_template(type_counter=None):
#     if(type_counter=='facts'):
#         return random.sample(['This is a fact','I know for a fact','There is an evidence'],1)[0]
#     elif(type_counter=='hypocrisy'):
#         return 'In contradiction'
#     elif(type_counter=='humour'):
#         return random.sample(['This is funny','You make me laugh','The funny part is'],1)[0]
#     elif(type_counter=='affiliation'):
#         return random.sample(['I also belong','I know','I have met'],1)[0]
#     elif(type_counter=='questions'):
#         return random.sample(['Do you really think that','Are you aware of','Where is the'],1)[0]
#     elif(type_counter=='denouncing'):
#         return random.sample(['Saying this is not right','Please do not say','This is hate speech'],1)[0]


# for most frequent prompts

def get_template(type_counter=None):
    if(type_counter=='facts'):
        return random.sample(['The vast majority of','Whilst a small number','They are here because','No it is not','Surely it is our'],1)[0]
    elif(type_counter=='hypocrisy'):
        return random.sample(['I do not think','What about recent school','Why have we built','How often have you','If you are really'],1)[0]
    elif(type_counter=='humour'):
        return random.sample(['Must be hard for','I had rather see them','Let us do the','I am really good','You make it sound'],1)[0]
    elif(type_counter=='affiliation'):
        return random.sample(['I am jewish and','I am a muslim','I am a christian','I am laughing so','I really want to'],1)[0]
    elif(type_counter=='questions'):
        return random.sample(['Have you thought about','How can you say','Why do you think','Where is your evidence','What does that even'],1)[0]
    elif(type_counter=='denouncing'):
        return random.sample(['This is not true','How can you say','Have won at what','Why is this a','What is truly cancerous'],1)[0]

    
    
# For manual prompts     
# def get_template(type_counter=None): 
#     if(type_counter=='facts'):
#         return random.sample(['This is a fact','I know for a fact','There is an evidence'],1)[0]
#     elif(type_counter=='hypocrisy'):
#         return 'In contradiction'
#     elif(type_counter=='humour'):
#         return random.sample(['This is funny','You make me laugh','The funny part is'],1)[0]
#     elif(type_counter=='affiliation'):
#         return random.sample(['I also belong','I know','I have met'],1)[0]
#     elif(type_counter=='questions'):
#         return random.sample(['Do you really think that','Are you aware of','Where is the'],1)[0]
#     elif(type_counter=='denouncing'):
#         return random.sample(['Saying this is not right','Please do not say','This is hate speech'],1)[0]

# to generate prompts based on cluster center

# def get_template(type_counter=None):
#     # to generate prompts based on cluster center
#     prompts_df=pd.read_csv("./Datasets/prompts.csv")
#     ls_facts=prompts_df[prompts_df["label"]=="facts"]["prompt"].values.tolist()
#     ls_hypocrisy=prompts_df[prompts_df["label"]=="hypocrisy"]["prompt"].values.tolist()
#     ls_humour=prompts_df[prompts_df["label"]=="humor"]["prompt"].values.tolist()
#     ls_affiliation=prompts_df[prompts_df["label"]=="affiliation"]["prompt"].values.tolist()
#     ls_questions=prompts_df[prompts_df["label"]=="question"]["prompt"].values.tolist()
#     ls_denouncing=prompts_df[prompts_df["label"]=="denouncing"]["prompt"].values.tolist()
    
#     if(type_counter=='facts'):
#         return random.sample(ls_facts,1)[0]
#     elif(type_counter=='hypocrisy'):
#         return random.sample(ls_hypocrisy,1)[0]
#     elif(type_counter=='humour'):
#         return random.sample(ls_humour,1)[0]
#     elif(type_counter=='affiliation'):
#         return random.sample(ls_affiliation,1)[0]
#     elif(type_counter=='questions'):
#         return random.sample(ls_questions,1)[0]
#     elif(type_counter=='denouncing'):
#         return random.sample(ls_denouncing,1)[0]


        



def generate_huggingface_method(params,hate_sentences,prepends,model,tokenizer,device,model_path,num_samples=10,type_counter=None):
    cntr = []
    types= []
    prepend = []
    model.eval()
    for step in tqdm(range(len(hate_sentences))):
        cntr_temp=[]
        type_temp=[]
        prepend_temp=[]
        for i in range(num_samples):
            # encode the new user input, add the eos_token and return a tensor in Pytorch
#             print(hate_sentences[step])
            type1=get_template(type_counter=type_counter)
            hate = ''
            if params['new_baseline']:
                type1 = "Counterspeech is a tactic of countering hate speech or misinformation by presenting an alternative narrative rather than with censorship of the offending speech."
                hate+= type1 + ' '
                
            if params['use_hatespeech']:
                hate+=hate_sentences[step]
                
            if params['use_prompts']:
                hate+=' '+type1

            phrase = np.random.choice(prepends, replace=False)
            if params['use_perpending']:
                hate+= ' '+phrase

#             print(hate)
                
            if params['use_hatespeech']:
                if params['new_baseline']:
                    input_hate = tokenizer.encode(type1 + ' ' + hate_sentences[step], truncation=True, max_length=params['max_input_length'], return_tensors='pt')
                else:
                    input_hate = tokenizer.encode(hate_sentences[step], truncation=True, max_length=params['max_input_length'], return_tensors='pt')
                input_hate=input_hate.to(device)
                
            if (params['use_hatespeech'] == False and params['use_prompts']==False):
                hate = ' '
#             print("Hate is ")
#             print(hate)
            input_ids = tokenizer.encode(hate,truncation=True,max_length=params['max_input_length'],return_tensors='pt') 
            input_ids=input_ids.to(device)
            ####### greedy_Decoding ######
            beam_outputs = model.generate(
                input_ids=input_ids, 
                pad_token_id         = tokenizer.eos_token_id,
                max_length           = params['max_generation_length']+len(input_ids[0]),
                min_length           = params['min_generation_length']+len(input_ids[0]),
                top_k                = params["k"],
                top_p                = params["p"],
                repetition_penalty   = params["repitition_penalty"],
                temperature          = params["temperature"],
                num_beams            = params['num_beams'], 
                do_sample            = params['sample'],
                no_repeat_ngram_size = params['no_repeat_ngram_size'],  
                early_stopping       = False
            )

            if params['use_hatespeech'] and ("google/flan-t5" not in model_path):
                reply_hate=tokenizer.decode(input_hate[0])
                reply = tokenizer.decode(beam_outputs[0])[len(reply_hate):].split('<|endoftext|>')[0]
            elif "google/flan-t5" in model_path:
                reply = tokenizer.decode(beam_outputs[0]).split('<|endoftext|>')[0]
                reply= type1+' '+reply
            else:
                reply = tokenizer.decode(beam_outputs[0]).split('<|endoftext|>')[0]
            
#             print(reply)
            
            cntr_temp.append(reply)
            type_temp.append(type1)
            prepend_temp.append(phrase)
        cntr.append(cntr_temp)
        types.append(type_temp)
        prepend.append(prepend_temp)
        if step>0 and step%20==0:
            if params['new_baseline']:
                print("***baseline prompt:***", type_temp[0])
            if params['use_hatespeech']:
                print("***hate:***", hate_sentences[step])
            if params['use_prompts']:
                print("***type***", type_temp[0])
            if params['use_perpending']:
                print("***prepend***", prepend_temp[0])
            print("***counter:***", cntr_temp[0])
            print("doing")
    return cntr,types,prepend


def main(params,model_path,dataset,gpu_id,num_samples,type_counter):
    path_models   = 'Saved_Models/Generator/'
    path_datasets = params['path_datasets'] 
    print('Dataset: ', path_datasets)

    if torch.cuda.is_available() and params['device']=='cuda':    
        # Tell PyTorch to use the GPU.    
            device = torch.device("cuda")
            ##### You can set the device manually if you have only one gpu
            ##### comment this line if you don't want to manually set the gpu
#            deviceID = [5]
            deviceID =get_gpu(model_path)
#            torch.cuda.set_device(1)
            ##### comment this line if you want to manually set the gpu
            #### required parameter is the gpu id
            torch.cuda.set_device(deviceID[0])

    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")
    
    
    test_path  = path_datasets+'/'+dataset+'/Test.json'
    cache_path = params['cache_path']
    
    if model_path == 'EleutherAI/gpt-j-6B':
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
    elif "falcon" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path,cache_dir=cache_path)
    if model_path == 'EleutherAI/gpt-j-6B':
#        config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
#         model = GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    elif model_path == 'EleutherAI/gpt-neo-2.7B':
        config = transformers.GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-2.7B")
        model = GPTNeoForCausalLM.from_pretrained("gustavecortal/gpt-neo-2.7B-8bit", low_cpu_mem_usage=True)
    elif "google/flan-t5" in model_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    elif "falcon" in model_path:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,
                                                     quantization_config=bnb_config,
                                                     device_map='auto')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path,cache_dir=cache_path)
    if "falcon" not in model_path:
        model.to(device)
#     model.to(device)
    model.eval()
    test  = pd.read_json(test_path)
    test_initiator_messages=[]
    
    hate_org_list=test['initiator_message'].tolist()
    counter_org_list=test['reply_message'].tolist()
    
    prepends = pd.read_json(params['prepend_path'])
    prepends = prepends[dataset].tolist()
    
    cntr_replies,types, prepend = generate_huggingface_method(params,hate_org_list,prepends,model,tokenizer,device,model_path,num_samples=num_samples,type_counter=type_counter)
    torch.cuda.empty_cache()
    
    dict_results={}
    dict_results['params']=params
    hate_counter_replies={}
    
    count=0
    
    print(len(hate_org_list),len(cntr_replies),len(types))
    
    for hate_org,counter,type1,prepend_tmp in zip(hate_org_list,cntr_replies,types,prepend):
        temp = {}
#         if params['use_hatespeech']:
        temp['hatespeech']=hate_org
#         if params['use_prompts']:
        temp['type']=type1
        temp['prepend']=prepend_tmp
        temp['counterspeech_model'] = counter

        hate_counter_replies[count]=temp
        count+=1    
    
    dict_results['samples']=hate_counter_replies  
    
    model_path_modified = "-".join(model_path.split('/')[-2:])
    print(model_path_modified)
    
    # using datetime module
    import datetime;
    # ct stores current time
    ct = datetime.datetime.now()
    
    # ts store timestamp of current time
    ts = ct.timestamp()
    ts = str(int(ts))
    
    write_path=params["save_path"] + model_path_modified
    if params['use_hatespeech']:
        write_in = write_path+"_on_"+dataset
        if(params['slur_removed']):
            write_in+="_without_slur"
        if params['use_prompts']:
            write_in+="_with_type_"+type_counter
        else:
            write_in+="_without_prompt"
        if params['use_perpending']:
            write_in+="_with_prepending"
        if params['new_baseline']:
            write_in+="_new_baseline"
    else:
        write_in = write_path+"_without_hatespeech_"+dataset
        if params['use_prompts']:
            write_in+="_with_type_"+type_counter
        else:
            write_in+="_without_prompt"

    if(params['generation_method']=='huggingface'):
            write_in+="_huggingface_base.json"   #### Output file name

    with open(write_in, 'w') as outfile:
         json.dump(dict_results, outfile,indent=4)

    
params = {
    'sep_token':'<|endoftext|>',
    'max_generation_length': 60,
    'min_generation_length':40,
    'max_input_length':256,
    'num_beams':1,
    'type_of_data':'implications', #implications, topic_argument
    'no_repeat_ngram_size':4,
    'repitition_penalty': 3.5,
    'k':50,
    'p':0.92,
    'sample':True,
    'temperature':1.2,
    'early_stopping':False,
    'dataset_hate':'CONAN',
    'save_path':'Results-Flan-t5-large/',
    'prepend_path':'./Datasets/frequent_phrases.json',
    'device': 'cuda',
    'batch_size':32,
    'cache_path':'../Saved_models/',
    'generation_method':'huggingface',
    'no_topic' : 3,
    'gpu_id':1,        ####      Set this as per availability
    'baseline':False,
}




if __name__ == "__main__":
    
    
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('model_path',
                           metavar='--p',
                           type=str,
                           help='location where model is saved')
    
    my_parser.add_argument('dataset',
                           metavar='--d',
                           type=str,
                           help='location where the dataset is saved')
    
    my_parser.add_argument('type',
                           metavar='--t',
                           type=str,
                           help='type of the counterspeech to be generated')
    
    my_parser.add_argument('--hs', help = 'Whether to use hatespeech', action = 'store_true')
    my_parser.add_argument('--prompt', help = 'Whether to use prompts', action = 'store_true')
    my_parser.add_argument('--phrase', help = 'Whether to use phrases', action = 'store_true')
    my_parser.add_argument('--noslur', help = 'Remove slur from hatespeech', action = 'store_true')
    my_parser.add_argument('--new_baseline', help = 'For new baselines', action = 'store_true')

    
    args = my_parser.parse_args()
    saved_path= args.model_path
    dataset=args.dataset
    type_counter=args.type
    num_samples=3
    model=saved_path
    print(model,dataset)

    params['use_hatespeech'] = args.hs
    params['use_prompts'] = args.prompt
    params['use_perpending'] = args.phrase
    params['slur_removed'] = args.noslur
    params['new_baseline'] = args.new_baseline
    
    
    if params['slur_removed']:
        params['path_datasets']='Additional_datasets/Slur_removed'
    else:
        params['path_datasets']='./Datasets'

    main(params,model,dataset,params['gpu_id'],num_samples,type_counter)