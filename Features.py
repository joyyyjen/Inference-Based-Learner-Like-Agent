from utils import truncate_seq
import sys
import logging
from tqdm import tqdm
import math
import random
logging.getLogger().setLevel(logging.INFO)


class InputFeatures(object):
    def __init__(self, input_ids, input_mask,segment_ids, label_id, guid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.guid = guid


class InputMaskedFeatures(object):
    def __init__(self, indexed_tokens, segment_ids, masked_index, masked_label, label_index):
        self.indexed_tokens = indexed_tokens
        self.segment_ids = segment_ids
        self.masked_index = masked_index
        self.masked_label = masked_label
        self.label_index = label_index


def truncateSeqMultiPair(tokens_list,target_indices,max_length,indexer):

    list_name = ['list'+str(i) for i in range(1,len(tokens_list))]
    list_name.append('hypothesis')
    
    list_to_index = {
        key: [val0[0:-1],val1,len(val0[0:-1])] 
        for key,val0,val1 in zip(list_name,tokens_list,target_indices)
        }
    
    list_to_index = indexer.truncate(list_to_index,max_length)
    return list_to_index


def convert_instances_to_features_entailment(instances,label_list,max_seq_length,tokenizer,indexer,instance_dict,
                                  set_type=None,dropout=None):
    label_map = {label:index for index,label in enumerate(label_list)}
    
    features = []
    for ex_index, instance in enumerate(tqdm(instances,desc="Feature Iteration")):
        wordpair = instance.word[0:2]
        wordpair_name = "|".join(wordpair)
        word_map = {word:index for index,word in enumerate(wordpair)}
        guid = instance.guid.split('-') # text_a = guid[0], text_b = guid[1], counter = guid[2]
        
       
        tokens_a = instance_dict[wordpair_name][guid[0]][1]
        tmp_index = instance_dict[wordpair_name][guid[0]][2] 
        target_index_a = tmp_index if type(tmp_index) == int else tmp_index[0]
        
        
        tokens_b = instance_dict[wordpair_name][guid[1]][1]
        tmp_index = instance_dict[wordpair_name][guid[1]][2] 
        target_index_b = tmp_index if type(tmp_index) == int else tmp_index[0]

        tokens_list = []
        tokens_list.append(tokens_a)
        tokens_list.append(tokens_b)
        target_indices = [target_index_a,target_index_b]
        list_to_index = truncateSeqMultiPair(tokens_list,target_indices,max_seq_length -3,indexer)
            
        tokens_a = []
        #if ex_index ==0:
        #    print(list_to_index)
        for name in list_to_index:
            if name != 'hypothesis':
                tokens_a.extend(list_to_index[name][0])
            
        tokens_b = list_to_index['hypothesis'][0]
        
        
        #truncate_seq_pair(target_index_a,tokens_a,target_index_b,tokens_b,max_seq_length-3)
        
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b)+1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)
        
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        
        if dropout != None: 
            input_id = input_ids
            indices = [ i for i,x in enumerate(input_id) if x == 102 ]
            premise = input_id[:(indices[0]+1)]
            hypothesis = input_id[(indices[0]+1):(indices[1])]
            hyp_index = list(range(0,len(hypothesis)))
            num = math.floor(len(hyp_index)*0.2)
            #random_indices = random.sample(hyp_index,num)
            unk_id = tokenizer.convert_tokens_to_ids(["[UNK]"])
            for i in hyp_index:
                if set_type != 'dropout': 
                    value = [0,0,0,0,0,0,0,0,1,1]
                else:
                    value = [0,0,1,1,1,1,1,1,1,1]

                dice = random.sample(value,1)
                if dice[0] == 1:
                    hypothesis[i]=unk_id[0]

            padding = input_id[indices[1]:]

            input_ids = premise + hypothesis + padding
                
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        label_id = label_map.get(instance.label, -1)
        
        guid_ord = [ord(char) for char in instance.guid]
        
        try:
            assert len(guid_ord) <= 25
        except:
            logging.warning("GUID_ORD TOO LONG")
            print(guid_ord)
            sys.exit(1)
        while len(guid_ord) < 25:
            guid_ord.append(ord('@'))
            
        assert len(guid_ord) == 25
        
        #and set_type[0:5]=='train' 
        '''
        if ex_index < 5:
            print("***Example***")
            print("guid: %s" % (instance.guid))
            print("tokens: %s "% " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s (id = %d)" % (instance.label, label_id))
        '''
        #else:
        #    sys.exit(1)
        
        features.append(
                InputFeatures(input_ids=input_ids,
                    input_mask = input_mask,
                    segment_ids = segment_ids,
                    label_id = label_id,
                    guid = guid_ord
                ))
        #if len(features) == 22000: return features
    
    return features
    
def assert_FITB_guid_length(instance_guid,set_type,word_id,new_data = True):
    if new_data == False:
        if set_type[0:5]!='sente':
            guid = instance_guid.split('-')[2:9]
            abbr_guid = guid[0:3] if word_id ==0 else guid[3:6]

            abbr_guid = abbr_guid + [guid[-1]]
            abbr_guid = "-".join(abbr_guid)

            guid_ord = [ord(char) for char in abbr_guid]
            #if ex_index <20:
            #    print("abbr_guid:{} label_id:{} word_id:{}".format(abbr_guid,label_id,word_id))
            try:
                assert len(guid_ord) <= 27
            except:
                logging.warning("GUID_ORD TOO LONG")
                print(abbr_guid)
                print(guid_ord)
                sys.exit(1)
            while len(guid_ord) < 27:
                guid_ord.append(ord('@'))

            assert len(guid_ord) ==27
            
        else:
            guid = instance_guid.split('-')[2:9]
            abbr_guid = guid = '-'.join(guid)
            guid_ord = [ord(char) for char in guid]
            
            try:
                assert len(guid_ord) <= 30
            except:
                logging.warning("GUID_ORD TOO LONG")
                print(abbr_guid)
                print(guid_ord)
                sys.exit(1)
            while len(guid_ord) < 30:
                guid_ord.append(ord('@'))

            assert len(guid_ord) ==30
    else:
        if set_type[0:5]!='sente':
            guid = instance_guid.split('-')[2:9]
            abbr_guid = guid[0:3] if word_id ==0 else guid[3:6]

            abbr_guid = abbr_guid + [guid[-1]]
            abbr_guid = "-".join(abbr_guid)

            guid_ord = [ord(char) for char in abbr_guid]
            #if ex_index <20:
            #    print("abbr_guid:{} label_id:{} word_id:{}".format(abbr_guid,label_id,word_id))
            try:
                assert len(guid_ord) <= 51
            except:
                logging.warning("GUID_ORD TOO LONG")
                print(abbr_guid)
                print(guid_ord)
                print(len(guid_ord))
                sys.exit(1)
            while len(guid_ord) < 51:
                guid_ord.append(ord('@'))

            assert len(guid_ord) ==51
            
            
        else:
            guid = instance_guid.split('-')[2:]
            guid = "-".join(guid)
            guid_ord = [ord(char) for char in guid]
            
            try:
                assert len(guid_ord) <= 36
            except:
                logging.warning("GUID_ORD TOO LONG")
                print(guid)
                print(guid_ord)
                sys.exit(0)
            while len(guid_ord) < 36:
                guid_ord.append(ord('@'))

            assert len(guid_ord) ==36
            abbr_guid = guid
            
    return guid_ord, abbr_guid         
        
def convert_instances_to_masked_features(
    instances,label_list,max_seq_length,tokenizer,indexer,instance_dict,
    set_type=None,dropout=None):
    
    if set_type[0:5]=='train':
        logging.info("Convert to features")
        logging.info("Set Type:{}".format(set_type))
        logging.info("Dropout: {}".format(dropout))
    #print(len(random_value))
    
    label_map = {label:index for index,label in enumerate(label_list)}
    
    features = []
    for ex_index, instance in enumerate(tqdm(instances,desc="Feature Iteration")):
        wordpair = instance.word[0:2]
        word_map = {word:index for index,word in enumerate(wordpair)}
        x = 1 if instance.guid.split('-')[-1] != 'counter' else 2
        guid = instance.guid.split('-')[2:-x]
        
        hyp_guid = instance.guid.split('-')[-x] 
        tokens_list = []
        #tokens_a = []
        target_indices= []
        #tokens_a_indices =[]
        for idi in guid:
            #tokens_a.extend(instance_dict[idi][1])
            tokens_list.append(instance_dict[idi][1])
            index = instance_dict[idi][2]
            target_indices.append(index if type(index) ==int else index[0])
        tokens_b = None
        if instance.text_b:
            tokens_b = instance_dict[hyp_guid][1]
            tokens_b_index = instance_dict[hyp_guid][2]
            #print(tokens_b,tokens_b_index)
            
            tokens_b,tokens_b_target_index,original_b = indexer.replace_target_word_with_mask(
                                                                tokens_b,tokens_b_index)
            instance_dict[hyp_guid][2] = tokens_b_target_index
            
            tokens_list.append(tokens_b)
            target_indices.append(tokens_b_target_index)
            list_to_index = truncateSeqMultiPair(tokens_list,target_indices,max_seq_length -3,indexer)
            
            tokens_a = []
            
            for name in list_to_index:
                if name != 'hypothesis':
                    tokens_a.extend(list_to_index[name][0])
            
            tokens_b = list_to_index['hypothesis'][0]
            #masked_index = tokens_b.index('[MASK]')
            
            
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b)+1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)
        
        
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        
        masked_label = [-1] * len(input_ids)
         
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(masked_label) == max_seq_length
        

        label_id = label_map.get(instance.label,-1)
        
        word_id = word_map.get(instance.word[2],-1)
        # logging.info(label_list[label_id])
        label_tokens_id = tokenizer.convert_tokens_to_ids(instance.label)
        
        try:
            masked_index = input_ids.index(103)
        except:
            print(tokens)
            print(input_ids)
            sys.exit(1)
        # logging.info(label_tokens_id)
        
        masked_label[masked_index] = label_tokens_id
        logging.info(masked_label)
        sys.exit(1)
        #guid_ord, abbr_guid = assert_FITB_guid_length(instance.guid,set_type,word_id,new_data = True)
        
        #logging.info('set_type[0:5]={}'.format(set_type[0:5]))

        if ex_index < 5  and set_type[0:5]=='train':
            print("***Example***")
            print("guid: %s" % (instance.guid))
            print("tokens: %s "% " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s (id = %d)" % (instance.label, label_id))
        
        #else:
        #    sys.exit(1)
        features.append(
                InputMaskedFeatures(
                    indexed_tokens=input_ids,
                    segment_ids = segment_ids,
                    masked_index = masked_index,
                    masked_label = masked_label,
                    label_index= label_id
                ))
        #if len(features) == 22000: return features
    
    return features
    
def convert_instances_to_features(instances,label_list,max_seq_length,tokenizer,indexer,instance_dict,
                                  set_type=None,dropout=None):
    if set_type[0:5]=='train':
        logging.info("Convert to features")
        logging.info("Set Type:{}".format(set_type))
        logging.info("Dropout: {}".format(dropout))
    #print(len(random_value))
    
    label_map = {label:index for index,label in enumerate(label_list)}
    
    features = []
    for ex_index, instance in enumerate(tqdm(instances,desc="Feature Iteration")):
        wordpair = instance.word[0:2]
        word_map = {word:index for index,word in enumerate(wordpair)}
        x = 1 if instance.guid.split('-')[-1] != 'counter' else 2
        guid = instance.guid.split('-')[2:-x]
        
        hyp_guid = instance.guid.split('-')[-x] 
        tokens_list = []
        #tokens_a = []
        target_indices= []
        #tokens_a_indices =[]
        for idi in guid:
            #tokens_a.extend(instance_dict[idi][1])
            tokens_list.append(instance_dict[idi][1])
            index = instance_dict[idi][2]
            target_indices.append(index if type(index) ==int else index[0])
        tokens_b = None
        if instance.text_b:
            tokens_b = instance_dict[hyp_guid][1]
            tokens_b_index = instance_dict[hyp_guid][2]
            #print(tokens_b,tokens_b_index)
            
            tokens_b,tokens_b_target_index,original_b = indexer.replace_target_word_with_mask(
                                                                tokens_b,tokens_b_index)
            instance_dict[hyp_guid][2] = tokens_b_target_index
            
            tokens_list.append(tokens_b)
            target_indices.append(tokens_b_target_index)
            list_to_index = truncateSeqMultiPair(tokens_list,target_indices,max_seq_length -3,indexer)
            
            tokens_a = []
            
            for name in list_to_index:
                if name != 'hypothesis':
                    tokens_a.extend(list_to_index[name][0])
            
            tokens_b = list_to_index['hypothesis'][0]
            
            
            
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b)+1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)
        
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        
        if dropout != None: 
            input_id = input_ids
            indices = [ i for i,x in enumerate(input_id) if x == 102 ]
            premise = input_id[:(indices[0]+1)]
            hypothesis = input_id[(indices[0]+1):(indices[1])]
            hyp_index = list(range(0,len(hypothesis)))
            num = math.floor(len(hyp_index)*0.2)
            #random_indices = random.sample(hyp_index,num)
            unk_id = tokenizer.convert_tokens_to_ids(["[UNK]"])
            for i in hyp_index:
                if set_type != 'dropout': 
                    value = [0,0,0,0,0,0,0,0,1,1]
                else:
                    value = [0,0,1,1,1,1,1,1,1,1]

                dice  = random.sample(value,1)
                if dice[0] == 1:
                    hypothesis[i]=unk_id[0]

            padding = input_id[indices[1]:]

                #print(input_ids)

            input_ids = premise + hypothesis + padding
             
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map.get(instance.label,-1)
        word_id = word_map.get(instance.word[2],-1)
        
        guid_ord, abbr_guid = assert_FITB_guid_length(instance.guid,set_type,word_id,new_data = True)
        
        #logging.info('set_type[0:5]={}'.format(set_type[0:5]))

        if ex_index < 5  and set_type[0:5]=='train':
            print("***Example***")
            print("guid: %s" % (instance.guid))
            print("tokens: %s "% " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s (id = %d)" % (instance.label, label_id))
        
        #else:
        #    sys.exit(1)
        features.append(
                InputFeatures(input_ids=input_ids,
                    input_mask = input_mask,
                    segment_ids = segment_ids,
                    label_id = label_id,
                    guid = guid_ord
                ))
        #if len(features) == 22000: return features
    
    return features


""" 
Convert to Single Feature when word embedding is not sBert
Not used in current paper
"""
def convertSingleFeatures2(instances,label_list,max_seq_length,tokenizer,indexer,instance_dict,set_type=None,dropout=None):
    logging.info("Converting Single Features")
    logging.info("Set Type:{}".format(set_type))
    logging.info("Dropout:{}".format(dropout))
    
    label_map = {label:index for index,label in enumerate(label_list)}
    features = {}
    uniq_list = {}
    hyp_list = set()
    counter_list = set()
    ex_list = set()
    reverse_dictionary = {}
    truncate_history = []
    for ex_index,instance in enumerate(tqdm(instances,desc="Single Feature Iteration")):
        guid = instance.guid.split('-')[2:8]
        hyp_guid = instance.guid.split('-')[-1] if instance.guid.split('-')[-1] != 'counter' else instance.guid.split('-')[-2]
        #tokens_a = []
        tokens_list = []
        #tokens_a_indices = []
        target_indices = []
        for idi in guid:
            sent =instance_dict[idi][1]
            tokens_list.append(sent)
            #tokens_a.extend(instance_dict[idi][1][:-2])
            index = instance_dict[idi][2]
            target_indices.append(index if type(index)==int else index[0])
            
        #if ex_index == 0:
        #    print(tokens_a)
        tokens_b = instance_dict[hyp_guid][1]
        tokens_b_index = instance_dict[hyp_guid][2]
        if hyp_guid not in truncate_history:
            truncate_history.append(hyp_guid)
            tokens_b,tokens_b_target_index,original_b = indexer.replace_target_word_with_mask(
                tokens_b,tokens_b_index,mode=True)
            truncate_seq(tokens_b_target_index,tokens_b,max_seq_length-3,set_type=None)
            assert tokens_b_target_index >= 0 and tokens_b_target_index <= len(tokens_b)
        
        label_id = label_map.get(instance.label,-1)
        word_id = label_map.get(instance.word[2],-1)
        
        
        #A instance contains 6 exmples and 1 hypothesis, only three of them have the same target word with hypothesis. 
        #To find the three example and the hypothesis in text_a
        #instance word = (word_a,word_b,word_in_hypothesis) 
        
        split_index = 0 if instance.word[2] == instance.word[0] else 3

    
        tokens_a = tokens_list[split_index:split_index+3]
        tokens_a_target_indices = target_indices[split_index:split_index+3]
        
        guid = instance.guid.split('-')[2:8]
        
        guid = guid[split_index:split_index+3]

        #deal with hypothesis first
        
        hyp_tok =  ["[CLS]"] + tokens_b + ["[SEP]"]
        hyp_segment_ids = [0] * len(hyp_tok)
        
        hyp_input_ids = tokenizer.convert_tokens_to_ids(hyp_tok)
        hyp_input_mask = [1] * len(hyp_input_ids)

        padding = [0] * (max_seq_length - len(hyp_input_ids))
        hyp_input_ids += padding
        hyp_input_mask += padding
        hyp_segment_ids += padding
        hyp_label_id = label_map.get(instance.label,-1)

        assert len(hyp_input_ids) == max_seq_length
        assert len(hyp_input_mask) == max_seq_length
        assert len(hyp_segment_ids) == max_seq_length
        #hyp_guid = "{}-{}-{}".format("-".join(guid[0:2]),guid[-1] if guid[-1] !='counter' else guid[-2],'hyp')
        hyp_guid = instance.guid.split('-')[-1] if instance.guid.split('-')[-1] != 'counter' else instance.guid.split('-')[-2]
        
        hyp_feature = InputFeatures(input_ids=hyp_input_ids,
                        input_mask = hyp_input_mask,
                        segment_ids = hyp_segment_ids,
                        label_id = hyp_label_id,
                        guid = hyp_guid)
        
        if hyp_guid not in features:
            features[hyp_guid] = hyp_feature
            hyp_list.add(hyp_guid)
            reverse_dictionary[hyp_guid]=" ".join(tokens_b)
        id_index = 0
        
        feature_group = []
        
        if "-".join(guid) in uniq_list:
            continue
        else:
            uniq_list["-".join(guid)] = ex_index
            
        for tok, target_index_a in zip(tokens_a,tokens_a_target_indices):
            truncate_seq(target_index_a,tok,max_seq_length-3,set_type=None)
            tokens = ["[CLS]"] + tok + ["[SEP]"] 
            segment_ids = [0] * len(tokens)
        
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1] * len(input_ids)

            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            
            #ex_guid = "{}-{}-{}".format("-".join(guid[0:2]),guid[2+id_index],'ex')
            ex_guid = guid[id_index]
            ex_feature = InputFeatures(input_ids=input_ids,
                        input_mask = input_mask,
                        segment_ids = segment_ids,
                        label_id = label_id,
                        guid = ex_guid )
            
            if ex_guid not in features:
                features[ex_guid] = ex_feature
                if ex_guid[0:2] != '11' and ex_guid[0:2]!= '22':
                    counter_list.add(ex_guid)
                else:
                    ex_list.add(ex_guid)
                reverse_dictionary[ex_guid]=" ".join(tok)
            #features[hyp_guid] = hyp_feature
            if ex_index ==0:
                print(ex_guid)
            id_index +=1
    
    logging.info("num of hyp:{}".format(len(hyp_list)))
    logging.info("num of counter:{}".format(len(counter_list)))
    logging.info("num of normal:{}".format(len(ex_list)))        
    return features,reverse_dictionary,(hyp_list,counter_list,ex_list)
