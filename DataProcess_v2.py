"""DataProcessor for New Clean Wiki Data"""
import logging
import random
import os
from definition import (DATA_ROOT, GLOBAL_INDEX, WORD_TO_ID, WORD_MAP)
from utils import json_dump, json_load

logging.getLogger().setLevel(logging.INFO)


class InputInstance(object):
    def __init__(self, guid, text_a, text_b, label, word):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.word = word


class DataProcess(object):
    def __init__(self, output_dir, indexer):
        
        self.output_path = output_dir
        self.transform_mapping = {}
        self.example_ratio = 3
        self.question_ratio = 1
        self.indexer = indexer

        self.wordpair = None
        self.wordpair_str = None
        self.inflection_flag = None
        self.inflection_flag_verb = None
        self.wordpair_key = None

    def update_wordpair_info(self, wordpair):
        self.wordpair = wordpair
        self.wordpair_str = "|".join(self.wordpair)
        self.inflection_flag = True if (self.wordpair[0] in WORD_MAP 
                                        and WORD_MAP[self.wordpair[0]] != "verb") else False
        self.inflection_verb_flag =  True if (self.wordpair[0] in WORD_MAP 
                                              and WORD_MAP[self.wordpair[0]] == "verb") else False
        
        self.wordpair_key = GLOBAL_INDEX[self.wordpair_str]
        if self.inflection_verb_flag:
            self.generate_mapping_verb(wordpair)
        else:
            self.generate_mapping(wordpair)

    def get_multipair_instance(self, wordpair_list, args, set_type, setting=None):
        """ return a list of instances and instance dictionary given multi-wordpair setting
        function to gather instances from one or more than one wordpairs
        """
        instances = []
        instance_dict = {}

        for wordpair in wordpair_list:
            self.update_wordpair_info(wordpair)

            instance, instance_dict[self.wordpair_str] = self.get_instances(
                wordpair,
                args.multi, args.example_set_size, args.sentence_selection_size,
                set_type, setting, entailment=args.entailment,number=""
                #args.version[-1]
            )
            
            instances.extend(instance)
            if set_type not in ['sentence_selection', 'behavior_check', 'test'] and setting != 'word_choice':
                logging.info("shuffle instances")
                random.shuffle(instances)
            
        logging.info("instance length:{}".format(len(instances)))
        # logging.info("instance keys:{}".format(instance_dict.keys()))
        return instances, instance_dict

    def get_instances(self, wordpair, multi, example_set_size, sentence_selection_size, set_type,
                      setting=None, entailment=False,number=None):
        """
        set type option: train, dev, test, sentence_selection
        setting option: normal, reverse, double-reverse, default None
        multi: defalut False
        hypothesis dropout: default False 
        """

        if setting is not None:
            instance_type = set_type + '-' + setting
        else:
            instance_type = set_type

        # Multiple Premises Case
        if multi:
            from multiInstanceProcessor_v2 import MultiInstanceProcessor
            mip = MultiInstanceProcessor(self, wordpair, instance_type, 
                                         example_set_size, sentence_selection_size,number)
            
            # Entailment model, NOT sentence selection
            if entailment is True and instance_type != 'sentence_selection':
                return mip.create_multinstances(
                        self.read_data(set_type), entailment)

            # Sentence Selection
            elif instance_type == 'sentence_selection' and multi:
                return mip.create_multinstances_sentence_selection(
                        self.read_data(set_type),entailment)
            
            # FITB MODEL, not Sentence Selection
            else: 
                return mip.create_multinstances(
                        self.read_data(set_type))
        
        
            '''
            # FITB Model, Sentence selection 
            if instance_type == 'sentence_selection' and multi and not entailment:
                return mip.create_multinstances_sentence_selection(
                        self.read_data(set_type))
            '''
        # Single Premise Case
        #else:
        #    from singleInstanceProcessor import SingleInstanceProcessor
        #    sip = SingleInstanceProcessor(self, wordpair, dropout, seed, instance_type, sentence_selection_size)
        #    return sip.create_instances(
        #        self.read_data(self.wordpair_str, set_type))

    def generate_mapping(self, wordpair):
        logging.info("Generate Mappinig")
        
        output_path = os.path.join(self.output_path, 'cache')
        
        if os.path.exists(os.path.join(output_path,'mapping.json')):
            logging.info("Mapping exists")
            self.transform_mapping = self.get_transform_mapping(self.wordpair,self.wordpair_str)
            return 
        
        transform_mapping = {}
        for word, other_word in zip(wordpair, wordpair[::-1]):
            transform_mapping[word] = other_word
            if word in WORD_MAP:
                inflection = WORD_MAP[word]
                transform_mapping[inflection] = WORD_MAP[other_word]
        self.transform_mapping = transform_mapping
        
        
        
        json_dump(output_path, 'mapping.json', self.transform_mapping)
        
    
    def generate_mapping_verb(self,wordpair):
        import codecs
        import json
        
        word_map_path = os.path.join(DATA_ROOT,"inflection_set_verb.json")
        tmp_path = os.path.join(self.output_path, 'cache','mapping.json')
        if os.path.exists(tmp_path):
            self.transform_mapping = self.get_transform_mapping(self.wordpair,self.wordpair_str)
            return 
        
        with codecs.open(word_map_path,'r',encoding="utf-8") as infile:
            WORD_MAP = json.load(infile)
        transform_mapping = {}
        for word, other_word in zip(wordpair, wordpair[::-1]):
            transform_mapping[word] = other_word
            for word1_inflection, word2_inflection in zip(WORD_MAP[word],WORD_MAP[other_word]):
                transform_mapping[word1_inflection] = word2_inflection
                
        self.transform_mapping = transform_mapping
        output_path = os.path.join(self.output_path, 'cache')
        json_dump(output_path, 'mapping.json', self.transform_mapping)
    
    def get_training_data(self, wordpair_str):
        """
        to get training data from new preprocessed dataset
        """
        return self.read_data(wordpair_str, 'train')
    
    def get_dev_data(self, wordpair_str):
        return self.read_data(wordpair_str, 'dev')
    
    def get_test_data(self, wordpair_str):
        return self.read_data(wordpair_str, 'test')           
   
    
    def read_sentence_selection(self):
        all_sents = json_load(DATA_ROOT, 'wiki_sentences_all_info_parsed_0410.json')
        sentence_selection_dictionary = {}
        
        for index, word in enumerate(self.wordpair):
            if index == 0:
                prefix = '11'
            else:
                prefix = '22'
            
            sents = [item['rawSentence'] for item in all_sents[self.wordpair_str][word]]
            sents.sort()
            tmp_dict = {
                prefix+str(i):sent for i,sent in enumerate(sents)
            }
            sentence_selection_dictionary.update(tmp_dict)
        
        return sentence_selection_dictionary

    def read_data(self, set_type):
        wordpair = self.wordpair
        data = {
                wordpair[0]: [],
                wordpair[1]: []
                }
        

        wiki_data = json_load(DATA_ROOT, 'WIKI_DATASET_{}.json'.format(self.wordpair_str))

        for word in self.wordpair:
            if self.inflection_flag is True:
                word_id = str(WORD_TO_ID["{}_inflection".format(word)])
            else:
                word_id = str(WORD_TO_ID[word])
            
            if set_type == 'train':
                items = dict(list(wiki_data[self.wordpair_key][word_id].items())[0:4000])
                if len(items.keys()) != 4000:
                    logging.warning("{} data broke".format(self.wordpair_str))
                data[word] = items
            elif set_type == 'dev':
                data[word] = dict(list(wiki_data[self.wordpair_key][word_id].items())[4000:4500])
            elif set_type == 'test' or set_type == 'behavior_check':
                data[word] = dict(list(wiki_data[self.wordpair_key][word_id].items())[4500:5000])
            elif set_type == 'sentence_selection':
                data[word] = dict(list(wiki_data[self.wordpair_key][word_id].items())[4500:5000])
                sentence_selection_dictionary = self.read_sentence_selection()
                    
        
        
        if set_type == 'sentence_selection':
            
            return (sentence_selection_dictionary, data)
        else:
            new_data = self.split_data_to_example_question(wordpair, data)
            return new_data

    def split_data_to_example_question(self, wordpair, data):
        """--split data into example, question --#
        # example = 75% question = 15%
        """
        random.seed(4)
        
        splitted_data = {
            wordpair[0]: {
                str_type: {}
                for str_type in ['example', 'question']
            },
            wordpair[1]: {
                str_type: {}
                for str_type in ['example', 'question']
            }
        }
        
        total_ratio = self.example_ratio + self.question_ratio

        for word_index, word in enumerate(wordpair):
            curr_indice = list(data[word].keys())
            random.shuffle(curr_indice)
            number_of_example = round(len(curr_indice)/total_ratio) * self.example_ratio
            number_of_question = round(len(curr_indice)/total_ratio) * self.question_ratio
            
            logging.info("number of example:{}".format(number_of_example))
            logging.info("number of question:{}".format(number_of_question))
            example_indice = curr_indice[0:number_of_example]
            
            question_indice = curr_indice[number_of_example:]
            
            for i in example_indice:
                splitted_data[word]['example'][i] = data[word][i] 
            for i in question_indice:
                splitted_data[word]['question'][i] = data[word][i]
        
        
            
        return splitted_data

    @staticmethod
    def split_data_to_normal_counter(wordpair, data,set_type):
        
        """
        #--split example into normal, counter --#

        # normal:counter=2:1
        # 0.75 *2/3 = normal = 2000  #250 behavior_check
        # 0.75 *1/3 = counter = 1000 #125 behavior_check
        # 0.25 = question = 1000 #125
        """
        logging.info('split data set_type:{}'.format(set_type))
        word = wordpair[0]
        
        splitted_data = {
            wordpair[0]: {
                str_type: {}
                for str_type in ['normal', 'counter']
                },
            wordpair[1]: {
                str_type: {}
                for str_type in ['normal', 'counter']
            }
        }

        example_indice_a = list(data[wordpair[0]]['example'].keys())
        
        number_of_example = len(example_indice_a)

        for word in wordpair:

            example_indice = list(data[word]['example'].keys())

            if set_type[0] == 'train':
                # ====  FORCE ON COUNTER ====# 
                number_of_counter = int(number_of_example * 1/3)
                number_of_normal = int(number_of_example *2/3)
                # ====  FORCE OFF COUNTER ==== comment above, uncomment below # 
                #number_of_counter = 0
                
                #number_of_normal = int(number_of_example)
                
                # ==== END ==== #
                for counter_index in example_indice[:number_of_counter]:
                    splitted_data[word]['counter'][counter_index] = data[word]['example'][counter_index]
                for normal_index in example_indice[number_of_counter:]:
                    splitted_data[word]['normal'][normal_index] = data[word]['example'][normal_index]
                    
                try:
                    assert len(splitted_data[word]['normal'])==number_of_normal
                except:
                    logging.info(len(splitted_data[word]['normal']))
                assert len(splitted_data[word]['counter']) == number_of_counter
                

            elif set_type[1] == 'reverse' or set_type[1] == 'double_reverse':
                for counter_index in example_indice[:]:
                    splitted_data[word]['counter'][counter_index] = data[word]['example'][counter_index]
                    splitted_data[word]['normal'] = {}

            else:
                for normal_index in example_indice[:]:
                    splitted_data[word]['normal'][normal_index] = data[word]['example'][normal_index]
                    splitted_data[word]['counter'] = {}

        return splitted_data
    
    
    def get_transform_mapping(self,wordpair,wordpair_str):
        input_path = os.path.join(self.output_path,'cache')
        transform_mapping = json_load(input_path,'mapping.json')
        
        return transform_mapping

class CrossValidation(object):
    def k_folds(self,data,n_split=5):
        groups,partition_size = self.partition(len(data),n_split)
        for i in range(n_split):
            valid_index = groups[i]
            valid_instances = data[valid_index:valid_index + partition_size]
            others = set(list(range(n_split))) - set([i])
            train_index = [groups[k] for k in others]
            train_instances = self.combine(data,train_index,partition_size)

            yield train_instances,valid_instances
    @staticmethod
    def combine(data,train_index,partition_size):
        new_data = []
        for start in train_index:
            new_data.extend(data[start:start+partition_size])
        return new_data
    
    @staticmethod
    def partition(length,n_split):
        n_partitions = int(length/n_split)
        current = 0
        parti_indice = []
        for i in range(n_split):
            parti_indice.append(current)
            current = current + n_partitions
        return parti_indice,n_partitions 
