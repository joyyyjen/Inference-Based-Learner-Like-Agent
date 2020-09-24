import codecs
import pickle
import json
import os
import logging
logging.getLogger().setLevel(logging.INFO)

def output_history(history_file, info):
    if os.path.exists(history_file):
        with codecs.open(history_file, 'r', encoding='utf-8') as infile:
            history = json.load(infile)
    else:
        history = {}
    for key, value in info.items():
        if key not in history:
            history[key] = []
        history[key].append(value)
        
    with open(history_file, 'w', encoding = 'utf-8') as outfile:
        json.dump(history,outfile, indent=4)


def create_json_output_file(output_path,target_file):
    with codecs.open(os.path.join(output_path,target_file), 'w', encoding='utf-8') as outfile:
        json.dump({}, outfile, indent=4)


def json_dump(output_path, target_file,info):
    with codecs.open(os.path.join(output_path,target_file), 'w',encoding='utf-8') as infile:
        json.dump(info, infile, indent=4)


def json_load(output_path,target_file):
     
    with codecs.open(os.path.join(output_path,target_file),'r',encoding='utf-8') as infile:
        info = json.load(infile)
    return info

        
def pickle_dump(output_path,target_file,info):
    with codecs.open(os.path.join(output_path,target_file), 'wb') as handle:
        pickle.dump(info,handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(output_path,target_file):
    with codecs.open(os.path.join(output_path,target_file), 'rb') as handle:
        info = pickle.load(handle)
    return info


def truncate_seq(target_index_a,tokens_a,max_length,set_type=None):
    
    while True:
        total_length = len(tokens_a)
        if total_length <= max_length:
            break
        if target_index_a == len(tokens_a)-1:
            tokens_a.pop(0)
        elif target_index_a > math.floor(len(tokens_a)/2):
            tokens_a.pop(0)
        else:
            tokens_a.pop(1)
      
    return True


def truncate_seq_pair(target_index_a,tokens_a,target_index_b,tokens_b,max_length,set_type=None):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            if target_index_a == len(tokens_a)-1:
                tokens_a.pop(0)
            elif target_index_a > math.floor(len(tokens_a)/2):
                tokens_a.pop(0)
            else:
                tokens_a.pop(1)
        else:
            if target_index_b == len(tokens_b)-1:
                tokens_b.pop(0)
            elif target_index_b > math.floor(len(tokens_b)/2):
                tokens_b.pop(0)
            else:
                tokens_b.pop(1)
    return True


def merge_dicts(x, y,z):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    embedding_dict = x.copy()
    embedding_dict.update(y)
    embedding_dict.update(z)
    assert len(embedding_dict) == (len(x)+len(y) +len(z))
    return embedding_dict


def reverse_index_tokenize(reverse_index,tokenizer):
    reverse_index_dict = {}
    for guid in reverse_index:
        sent = reverse_index[guid]
        tokens = tokenizer.tokenize(sent)
        reverse_index_dict[guid] = (tokens,sent)
    return reverse_index_dict


""" randomize an ordered train features and keep the set form 
return random features for training
"""
def preprocessing_randomize(train_features):
    logging.info("random")
    chunks = [train_features[x:x+4] for x in range(0, len(train_features), 4)]
    random.shuffle(chunks)
    random_train_features = []
    for one_set in chunks:
        random.shuffle(one_set)
        random_train_features.extend(one_set)
    return random_train_features

""" given randomized logits and guid 
return x1,x2,y for marign ranking loss
"""
def retrieve_x1_x2_y(logits,guid):
    length = len(logits)
    indices = list(range(length))
    split_index = [i for i in indices if i%4 ==0]
    assert  len(split_index)%4 ==0
    y_label = []
    x1 = []
    x2 = []
    for index in split_index:
        current_combination_logits = logits[index:index+4]
        
        curr_comb_guid_quest_prefix = [i.split('-')[1][0:2] for i in guid[index:index+4]]
        
        w1_index =  curr_comb_guid_quest_prefix.index('11')         # x1
        w2_index =  curr_comb_guid_quest_prefix.index('22')         # x1_2
        counter_w1_index =  curr_comb_guid_quest_prefix.index('12') # x2
        counter_w2_index =  curr_comb_guid_quest_prefix.index('21') # x2_2
        
        for i_index in range(index,index+4):
            current_q_prefix = guid[i_index].split('-')[1][0:2]
            current_example_prefix = guid[i_index].split('-')[0][0:2]
            if current_example_prefix in ['11','22']:
                if current_q_prefix == '11':
                    x1.append(current_combination_logits[w1_index][0])
                    x2.append(current_combination_logits[counter_w1_index][0])
                    y_label.append(1)
                elif current_q_prefix == '22':
                    x1.append(current_combination_logits[w2_index][1])
                    x2.append(current_combination_logits[counter_w2_index][1])
                    y_label.append(1)
                elif current_q_prefix == '12':
                    x1.append(current_combination_logits[counter_w1_index][0])
                    x2.append(current_combination_logits[w1_index][0])
                    y_label.append(-1)
                elif current_q_prefix == '21':
                    x1.append(current_combination_logits[counter_w2_index][1])
                    x2.append(current_combination_logits[w2_index][1])
                    y_label.append(-1)
            elif current_example_prefix in ['12','21']:
                if current_q_prefix == '11':
                    x1.append(current_combination_logits[w1_index][0])
                    x2.append(current_combination_logits[counter_w1_index][0])
                    y_label.append(-1)
                elif current_q_prefix == '22':
                    x1.append(current_combination_logits[w2_index][1])
                    x2.append(current_combination_logits[counter_w2_index][1])
                    y_label.append(-1)
                elif current_q_prefix == '12':
                    x1.append(current_combination_logits[counter_w1_index][0])
                    x2.append(current_combination_logits[w1_index][0])
                    y_label.append(1)
                elif current_q_prefix == '21':
                    x1.append(current_combination_logits[counter_w2_index][1])
                    x2.append(current_combination_logits[w2_index][1])
                    y_label.append(1)
                    
    return x1,x2,y_label
