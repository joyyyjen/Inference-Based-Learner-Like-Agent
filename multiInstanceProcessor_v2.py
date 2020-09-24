"""MultiInstance Processor for New Clean Wiki Data"""
from definition import WORD_TO_ID, WORD_MAP, GLOBAL_INDEX
from DataProcess_v2 import InputInstance
from utils import pickle_dump, merge_dicts, pickle_load
from tqdm import tqdm
from itertools import combinations
import random
import re
import os
import json
import codecs
import logging
logging.getLogger().setLevel(logging.INFO)



class MultiInstanceProcessor(object):
    def __init__(self,processor,wordpair,instance_type,example_set_size,sentence_selection_size=None,number=None):
        
        self.example_dict = {'all':{},'embedding':{}}
        self.hyp_dict = {'all':{},'embedding':{}}
        self.counter_dict = {'all':{},'embedding':{}}
        self.sentence_selection_dict = {}
        self.sentence_selection_size = sentence_selection_size
        self.example_set_size = example_set_size
        self.processor = processor
        self.number = number
        
        self.dropout = False #self.dropout = dropout #should delete this portion
        
        self.seed = 4
        self.wordpair = wordpair
        self.wordpair_str = "|".join(self.wordpair)
        self.wordpair_key = GLOBAL_INDEX[self.wordpair_str]
        inflection_flag = True if (self.wordpair[1] in WORD_MAP and WORD_MAP[self.wordpair[1]]!= "verb") else False
        logging.info("inflection flag:{} word1:{}".format(inflection_flag,self.wordpair[0]))
        self.word_id = []
        self.word_id.append(
            WORD_TO_ID["{}_inflection".format(self.wordpair[0]) 
                       if inflection_flag is True else self.wordpair[0]])
        self.word_id.append(
            WORD_TO_ID["{}_inflection".format(self.wordpair[1]) 
                       if inflection_flag is True else self.wordpair[1]])
        instance_type = instance_type.split('-')
        if len(instance_type) ==1:
            instance_type.append("")
        self.set_type = instance_type
        
        
        self.verbose = True if self.set_type[0] =='train' else False
        self.example_ratio = 3
        self.question_ratio = 1
        
    
    @staticmethod
    def sample_x_sentences(example_indice,x):
        """ randomly pick x sentence for an instance"""
        random.seed(4)
        example_length = len(example_indice)
        logging.warning("sample {} sentences example indice length:{}"
                        .format(x,example_length))
        num = set()
        index_combination = []
        while len(num)!= example_length or len(index_combination)!=example_length:
            tmp = random.sample(example_indice,x)
            if all(i in num for i in tmp) and len(num)!=example_length:
                continue
            else:
                index_combination.append(tmp)
                for i in tmp:
                    num.add(i)
        return index_combination
   
    
    def save_sentence(self,example_a_guid,example_a,sent_type,embedding):
        output_path = os.path.join(self.processor.output_path,'{}_example_sentence.txt'.format(self.set_type[0]))
        file = codecs.open(output_path,'a+',encoding='utf-8')
        
        tmp_dict = {
            guid_i:example_i for guid_i,example_i in zip(example_a_guid,example_a)
        }
        
        #logging.info(tmp_dict)
        if sent_type == 'example':
           
            if embedding:
                self.example_dict['embedding'].update(tmp_dict)
            else:
                self.example_dict['all'].update(tmp_dict)
            json.dump(tmp_dict,file,indent = 4)
        if sent_type == 'counter':
            
            if embedding:
                self.counter_dict['embedding'].update(tmp_dict)
            else:
                self.counter_dict['all'].update(tmp_dict)
            json.dump(tmp_dict,file,indent = 4)
        if sent_type == 'hyp':
           
            self.hyp_dict['embedding'].update(tmp_dict)
            self.hyp_dict['all'].update(tmp_dict)

            json.dump(tmp_dict,file,indent = 4)
            
    def create_example_instance(self,a_indice,b_indice,data,counter = None):
        """ concat example instance """
        
        
        example_a, example_b = [],[]
        example_a_guid, example_b_guid = [],[]
        
        indice_a,indice_b = a_indice.pop(0),b_indice.pop(0)
        
        assert len(indice_a) == self.example_set_size
        
        flag_index = 0 
        for word in self.wordpair:
            for text_index in indice_a if flag_index < self.example_set_size else indice_b:
               
                if counter is not None:
                    example = data[word]['counter'][text_index]['rawSentence'].strip('\n')

                    example = self.swap_word(data[word]['counter'][text_index]['targetWord'],
                                              example,)
                    word_a_prefix = '12'
                    word_b_prefix = '21'
                else:
                    example = data[word]['normal'][text_index]['rawSentence'].strip('\n')
                    word_a_prefix = '11'
                    word_b_prefix = '22'

                if flag_index < self.example_set_size:
                    example_a.append(example)
                    example_a_guid.append(word_a_prefix+str(text_index))
                else:
                    example_b.append(example)
                    example_b_guid.append(word_b_prefix+str(text_index))
                flag_index +=1
        
        
        assert len(example_a_guid) == self.example_set_size
        assert len(example_b_guid) == self.example_set_size
        if counter is not None:
            text_a = example_b + example_a
            guid = example_b_guid + example_a_guid
            self.save_sentence(example_b_guid,example_b,'counter',False)
            self.save_sentence(example_a_guid,example_a,'counter',False)
            
        else:
            text_a = example_a + example_b
            guid = example_a_guid + example_b_guid
            self.save_sentence(example_b_guid,example_b,'example',False)
            self.save_sentence(example_a_guid,example_a,'example',False)
        
        return text_a,guid
    
    def swap_word(self,target,string):
        """ make counter text for counter instance"""
        transform_mapping = self.processor.get_transform_mapping(self.wordpair,self.wordpair_str)
        counter_word = transform_mapping[target]
        counter_text = re.sub(r"\b%s\b" % target,counter_word,string)
        return counter_text
    
    def guid(self,word,text_a_guid,text_b_guid,counter):
        """ return guid format
        Input: set_type, word, indices of sentence contains word1, indices of sentences contain word2,
               counter or not
        """
        set_type = "_".join(self.set_type) if self.set_type[1] !="" else self.set_type[0] 
        guid = "%s-%s-%s-%s" % (set_type,word,text_a_guid,text_b_guid)
        if counter:
            guid = guid+'-counter'
        return guid
    
    def create_entailment_InputInstance(self,target_words,text_b_examples,indice,
                                        text_a,text_a_guid,counter_instance):
        
        instances = []
        
        text_b_guid_w1 = '11'+str(indice[0])
        text_b_guid_w2 = '22'+str(indice[1])
        
        counter_text_b_w1 = self.swap_word(target_words[0],text_b_examples[0])
        counter_text_b_w2 = self.swap_word(target_words[1],text_b_examples[1])

        counter_text_b_guid_w1 = '12'+str(indice[0])
        counter_text_b_guid_w2 = '21'+str(indice[1])
        
        if self.set_type[0] != 'sentence_selection':

            self.save_sentence([counter_text_b_guid_w1],[counter_text_b_w1],'hyp',False)      
            self.save_sentence([counter_text_b_guid_w2],[counter_text_b_w2],'hyp',False)
        else:
            
            self.sentence_selection_dict[counter_text_b_guid_w1] = counter_text_b_w1
            self.sentence_selection_dict[counter_text_b_guid_w2] = counter_text_b_w2

        pair_count = 0
        
        x = self.example_set_size
        

        for pointer,text_i in enumerate(zip(text_a,text_a_guid)):
            if (pointer < x and counter_instance == False) or (pointer >= x and counter_instance == True):
                tmp_instances = [InputInstance(
                    guid = text_i[1] +'-'+ text_b[0],
                    text_a = text_i[0],
                    text_b = text_b[1],
                    label = 'True' if (flag ==0 and not counter_instance 
                                       or flag ==2 and counter_instance) else 'False',
                    word = (self.wordpair[0],
                            self.wordpair[1],
                            self.wordpair[0] if (text_b[0][0:2]=='11' 
                                                 or text_b[0][0:2] =='21') else self.wordpair[1])
                    ) for flag,text_b in 
                        enumerate(zip([text_b_guid_w1,text_b_guid_w2,
                                       counter_text_b_guid_w1,counter_text_b_guid_w2],
                                      [text_b_examples[0],text_b_examples[1],
                                       counter_text_b_w1,counter_text_b_w2]))  ]

                pair_count +=4
                
                
            elif ((pointer  >= x and counter_instance == False) 
                   or 
                  (pointer < x and counter_instance == True)
                 ):
                
                tmp_instances = [InputInstance(
                    guid = text_i[1] + '-'+text_b[0],
                    text_a = text_i[0],
                    text_b = text_b[1],
                    label = 'True' if (flag ==1 and not counter_instance 
                                       or flag== x 
                                       and counter_instance) else 'False',
                    word = (self.wordpair[0],
                            self.wordpair[1],
                            self.wordpair[0] if (text_b[0][0:2]=='11' or text_b[0][0:2] =='21') else self.wordpair[1])
                    ) for flag,text_b in 
                        enumerate(zip([text_b_guid_w1,text_b_guid_w2,
                                       counter_text_b_guid_w1,counter_text_b_guid_w2],
                                      [text_b_examples[0],text_b_examples[1],
                                       counter_text_b_w1,counter_text_b_w2]))  ]
                pair_count +=4
                
            instances.extend(tmp_instances)
        return instances
    
    
    def create_1_pairs_InputInstance(self,text_a,text_a_guid,text_b_guid_w1,text_b_guid_w2,
                                                 text_b_w1,text_b_w2, counter_instance):
        instances = []
        pairs = 0
        for word,text_b_guid in zip(self.wordpair,[text_b_guid_w1,text_b_guid_w2]):
            a = text_a_guid[:self.example_set_size]
            b = text_a_guid[self.example_set_size:]
            text_a_a = text_a[:self.example_set_size]
            text_a_b = text_a[self.example_set_size:]
            
            m = len(a)-1
            guids = []
            while m >= 0:
                tmp_guid = self.guid(word,a[m]+'-'+b[m],text_b_guid,counter_instance)
                m -=1
                guids.append(tmp_guid)
            
            
            #sys.exit(1)
            if counter_instance and self.set_type[1] != 'reverse':
                text_b = {
                    guid:(text,word,label) 
                    for guid,text,word,label in zip(
                        guids,[text_b_w1,text_b_w2],self.wordpair,self.wordpair[::-1])
                    }
            else:
                text_b = {
                    guid:(text,word,word) 
                    for guid,text,word in zip(
                        guids,[text_b_w1,text_b_w2],self.wordpair)

                }
            
            m = len(a)-1
            tmp_instances = []
            while m >= 0:
                for guid in guids:
                    tmp_instances.append(
                        InputInstance(
                        guid = guid, text_a = text_a_a[m]+'\@'+text_a_b[m], text_b = text_b[guid][0],
                        label = text_b[guid][2],
                        word = (self.wordpair[0],self.wordpair[1],text_b[guid][1])
                        )
                    )
                m-=1
                pairs+=1
            
                    
            instances.extend(tmp_instances)
        #logging.info(pairs)
        pairs = 0
        return instances
            
    def create_regular_three_pairs_InputInstance(self,text_a,text_a_guid,text_b_guid_w1,text_b_guid_w2,
                                                 text_b_w1,text_b_w2, counter_instance):
                
        instances = []
        
        guids = [self.guid(word,"-".join(text_a_guid),text_b_guid,counter_instance) 
                         for word,text_b_guid in zip(self.wordpair,[text_b_guid_w1,text_b_guid_w2])]
        
        # ----- Add to Input Instance ---- #
        # during training and double-reverse, both label and text_a are reversed
        if counter_instance and self.set_type[1] != 'reverse':
            text_b = {
                guid:(text,word,label) 
                for guid,text,word,label in zip(guids,[text_b_w1,text_b_w2],
                                                self.wordpair,self.wordpair[::-1])
                }
        else:
            text_b = {
                guid:(text,word,word) 
                for guid,text,word in zip(guids,[text_b_w1,text_b_w2],self.wordpair)

            }
        
        tmp_instances = [InputInstance(
            guid = guid, text_a = "\@".join(text_a), text_b = text_b[guid][0],
            label = text_b[guid][2],
            word = (self.wordpair[0],self.wordpair[1],text_b[guid][1]))
            for guid in guids ]
        
        instances.extend(tmp_instances)
        
        return instances

    def create_multinstances(self,data,entailment=False):
        
        example_data = self.processor.split_data_to_normal_counter(self.wordpair,data,self.set_type)
        
        # here a,b refer to word1 and word2
        normal_a_indice = self.sample_x_sentences(
            list(example_data[self.wordpair[0]]['normal'].keys()),
            self.example_set_size)
        normal_b_indice = self.sample_x_sentences(
            list(example_data[self.wordpair[1]]['normal'].keys()),
            self.example_set_size)
        counter_a_indice = self.sample_x_sentences(
            list(example_data[self.wordpair[0]]['counter'].keys()),
            self.example_set_size)
        counter_b_indice = self.sample_x_sentences(
            list(example_data[self.wordpair[1]]['counter'].keys()),
            self.example_set_size)
        
        logging.info("normal_a_indice length:{}".format(len(normal_a_indice)))
        logging.info("normal_b_indice length:{}".format(len(normal_b_indice)))
        logging.info("counter_a_indice length:{}".format(len(counter_a_indice)))
        logging.info("counter_b_indice length:{}".format(len(counter_b_indice)))
        
        total_length = len(example_data[self.wordpair[0]]['normal']) + len(example_data[self.wordpair[0]]['counter'])
        logging.info("total length:{}".format(total_length))
        
        # ========== grouping example sentences ========== #
        example_instances = []
        normal_count = 0
        counter_count = 0
        for example_count in range(total_length):
            
            #---- Counter Setting ----#
            counter_instance = True if example_count % 3 == 0 else False
            if self.set_type[1] == 'reverse' or self.set_type[1] == 'double_reverse':
                counter_instance = True
            elif self.set_type[0] !='train' or self.dropout == True:
                counter_instance = False
            
            #counter_instance = False ## Turn off Counter Instance
            
            if counter_instance == True:
                counter_text_a,counter_guid = self.create_example_instance(
                    counter_a_indice,counter_b_indice,example_data,counter=True)
                example_instances.append((counter_guid,counter_text_a))
                counter_count +=1
            else:
                normal_text_a,normal_guid = self.create_example_instance(
                    normal_a_indice,normal_b_indice,example_data)
                example_instances.append((normal_guid,normal_text_a))
                normal_count +=1
                
            # logging.info("\r counter_count:{}\t normal_count:{}     \r".format(counter_count,normal_count))

            
        # ========== concat example with questions =========== #
        question_a_data = data[self.wordpair[0]]['question']
        question_b_data = data[self.wordpair[1]]['question']
        
        print(len(question_a_data))
        print(data.keys())
        print(data[self.wordpair[0]].keys())
        example_list = []
        instances = []
        number_of_question_per_id = 10 if not entailment else 5
        
        
        question_a_indice = list(question_a_data.keys())
        question_a_length = len(question_a_indice)
        question_a_indice.extend(question_a_indice[0:number_of_question_per_id-1])
        text_b_word_a_indice = [question_a_indice[i:i+number_of_question_per_id] 
                                    for i in range(len(question_a_indice)-(number_of_question_per_id -1))]
        
        question_b_indice = list(question_b_data.keys())
        question_b_length = len(question_b_indice)
        question_b_indice.extend(question_b_indice[0:number_of_question_per_id-1])
        text_b_word_b_indice = [question_b_indice[i:i+number_of_question_per_id] 
                                    for i in range(len(question_b_indice)-(number_of_question_per_id -1))]
        
        logging.info("text b word a length:{}".format(len(text_b_word_a_indice)))
        logging.info("text b word b length:{}".format(len(text_b_word_b_indice)))
        
        counter_instance_count = 0
        normal_instance_count = 0
        
        for ex_index,ex_instance in enumerate(tqdm(example_instances,desc='Instance Iteration')):
            text_a_guid = ex_instance[0]
            
            text_a = ex_instance[1]
            
            prefix = text_a_guid[0][0:2]
            if prefix =='21': 
                counter_instance = True
                counter_instance_count +=1
            else:
                counter_instance = False
                normal_instance_count +=1  
            
            text_b_word_a_index = text_b_word_a_indice.pop(0)
            text_b_word_a_indice.append(text_b_word_a_index)
            text_b_word_b_index = text_b_word_b_indice.pop(0)
            text_b_word_b_indice.append(text_b_word_b_index)
            
            for a,b in zip(text_b_word_a_index,text_b_word_b_index):
                text_b_w1 = question_a_data[a]['rawSentence'].strip('\n')
                text_b_w2 = question_b_data[b]['rawSentence'].strip('\n')
                
                
                text_b_guid_w1 = '11'+str(a)
                text_b_guid_w2 = '22'+str(b)
                
                self.save_sentence([text_b_guid_w1],[text_b_w1],'hyp',False)      
                self.save_sentence([text_b_guid_w2],[text_b_w2],'hyp',False)
                
                if entailment:
                    
                    target_words = [question_a_data[a]['targetWord'],question_b_data[b]['targetWord']]
                    text_b_examples = [text_b_w1,text_b_w2]
                    indice = [a,b]
                    tmp_instances = self.create_entailment_InputInstance(target_words,
                        text_b_examples,indice,
                        text_a,text_a_guid,
                        counter_instance)
                    instances.extend(tmp_instances)
                    
                else:
                    tmp_instances = self.create_regular_three_pairs_InputInstance(
                        text_a,text_a_guid,
                        text_b_guid_w1,text_b_guid_w2,
                        text_b_w1,text_b_w2,
                        counter_instance)
                    instances.extend(tmp_instances)
            
        
        logging.info("Final Length of instances:{}".format(len(instances)))
        
        if self.set_type[0] == 'train' and self.dropout == False:
            split_num = 12 if entailment else 1
            expect_length = (total_length *number_of_question_per_id*2)*split_num
            logging.info("normal instance:{}".format(normal_instance_count))
            logging.info("counter instance:{}".format(counter_instance_count))
            logging.info("Train without Dropout Expect Length of instances:{}".format(expect_length))
            assert len(instances) == expect_length

        elif self.set_type[0] == 'dev' or self.set_type[0] == 'test' or self.set_type[1] == 'normal' or self.dropout == True:
            #assert normal_instance_count == example_length *10
            logging.info('total length:{}'.format(total_length))
            logging.info('num index:{}'.format(number_of_question_per_id))
            split_num = 12 if entailment else 1
            expect_length = (total_length *number_of_question_per_id*2)*split_num
            logging.info("{} Expect Length of instances:{}".format(self.set_type[0],expect_length))
            try:
                assert len(instances) == expect_length
            except:
                print()
        elif self.set_type[1] == 'reverse' or self.set_type[1]=='double_reverse':
                #assert counter_instance_count == example_length *10
            split_num = 12 if entailment else 1
            expect_length = (total_length *number_of_question_per_id*2)*split_num
            logging.info("{} Expect Length of instances:{}".format(self.set_type[1],expect_length))
            assert len(instances) == expect_length
        
        logging.info('question dict length:{}'.format(len(self.hyp_dict['all'])))
        logging.info('normal example dict length:{}'.format(len(self.example_dict['all'])))
        logging.info('counter example dict length:{}'.format(len(self.counter_dict['all'])))
        if self.set_type[0] == 'train':
            
            pickle_dump(self.processor.output_path,'train_hyp.pickle',self.hyp_dict['all'])
            pickle_dump(self.processor.output_path,'train_example.pickle',self.example_dict['all'])    
            pickle_dump(self.processor.output_path,'train_counter.pickle',self.counter_dict['all']) 
            
            
            instance_dict = merge_dicts(self.example_dict['all'],self.counter_dict['all'],self.hyp_dict['all'])
            assert len(instance_dict) == (len(self.example_dict['all'])+
                                          len(self.counter_dict['all'])+
                                          len(self.hyp_dict['all']))
            self.example_dict['all'], self.counter_dict['all'],self.hyp_dict['all'] = None, None, None
            return (instances,instance_dict)
        else:
                
            instance_dict = merge_dicts(self.example_dict['all'],self.counter_dict['all'],self.hyp_dict['all'])
            self.example_dict['all'], self.counter_dict['all'], self.hyp_dict['all'] = None, None, None
            
            return instances,instance_dict
    
    
    def sentence_selection_combination(self, indice, prefix):
        
        example = []
        
        for index in indice:
            raw_sentence = self.sentence_selection_dict[prefix+str(index)]
            example.append(raw_sentence)
            #if prefix + str(index) not in self.sentence_selection_dict:
            #    self.sentence_selection_dict[prefix+str(index)] = raw_sentence
        return example 
    
    def create_counter_question(self,target_word, other_word, sentence):
        transform_mapping = self.processor.get_transform_mapping(self.wordpair)
        # find index
        index = self.processor.indexer.findIndexSingleCase(target_word, sentence.split())
        counter_text = sentence.split()
        if counter_text[index] in transform_mapping[target_word]:
            counter_text[index] = transform_mapping[target_word][counter_text[index]]
        else:
            counter_text_b_w1[index_b_w1] = other_word
        counter_text = " ".join(counter_text)

    def create_multinstances_sentence_selection(self, data, entailment):
        
        self.sentence_selection_dict , question_data = data
        ex_index= [i for i in range(10)]
        num = 3
        sentence_selection_size = self.sentence_selection_size # original 5 for each word
        logging.info("sentence selection size:{}".format(sentence_selection_size))

        if entailment: 
            #question_a_indice, question_b_indice = self.sentence_selection_entailment_exception(
            #    question_data,sentence_selection_size)
            
            question_a_indice, question_b_indice = self.sentence_selection_entailment_group_compair(
                question_data,sentence_selection_size)
            
            
        else:
            question_a_indice = list(question_data[self.wordpair[0]].keys())
            question_a_indice = question_a_indice[:sentence_selection_size]
            

            question_b_indice = list(question_data[self.wordpair[1]].keys())
            question_b_indice = question_b_indice[:sentence_selection_size]


        instances = []
        example_b_dictinary = {}
        for indices_1 in tqdm(combinations(ex_index,num)):
            example_a = self.sentence_selection_combination(indices_1,'11')
            
            for indices_2 in tqdm(combinations(ex_index,num)):
                # obtain example_b
                if indices_2 not in example_b_dictinary:
                    example_b_dictinary[indices_2] = self.sentence_selection_combination(indices_2, '22')
                    
                example_b = example_b_dictinary[indices_2]
                
                #text_a = "".join(example_a) + "".join(example_b)
                text_a = example_a + example_b
                text_a_guid_w1 = ["11"+str(index) for index in indices_1]
                text_a_guid_w2 = ["22"+str(index) for index in indices_2]
                text_a_guid = text_a_guid_w1 + text_a_guid_w2
                #text_a_guid_w1 = '-'.join(["11"+str(index) for index in indices_1]) 
                #text_a_guid_w2 = '-'.join(["22"+str(index) for index in indices_2])
                #text_a_guid = text_a_guid_w1 +'-'+text_a_guid_w2
                
                for a,b in zip(question_a_indice,question_b_indice):
                    if '11'+str(a) in self.hyp_dict:
                        text_b_w1 = self.hyp_dict['all']['11'+str(a)]
                        text_b_w2 = self.hyp_dict['all']['22'+str(b)]

                        text_b_guid_w1 = '11'+str(a)
                        text_b_guid_w2 = '22'+str(b)
                    
                    else:
                        text_b_w1 = question_data[self.wordpair[0]][a]['rawSentence'].strip('\n')
                        text_b_w2 = question_data[self.wordpair[1]][b]['rawSentence'].strip('\n')
                        text_b_guid_w1 = '11'+str(a)
                        text_b_guid_w2 = '22'+str(b)
                    
                    if text_b_guid_w1 not in self.sentence_selection_dict:
                        self.sentence_selection_dict[text_b_guid_w1] = text_b_w1
                        self.sentence_selection_dict[text_b_guid_w2] = text_b_w2

                    if not entailment:
                        tmp_instances = self.create_regular_three_pairs_InputInstance(
                        text_a,text_a_guid,
                        text_b_guid_w1,text_b_guid_w2,
                        text_b_w1,text_b_w2,
                        False)
                        instances.extend(tmp_instances)
                    
                    else:
                       
                        target_words = [
                            question_data[self.wordpair[0]][a]['targetWord'],
                            question_data[self.wordpair[1]][b]['targetWord']]
                        text_b_examples = [text_b_w1,text_b_w2]
                        indice = [a,b]
                        tmp_instances = self.create_entailment_InputInstance(target_words,
                            text_b_examples,indice,
                            text_a,text_a_guid,
                            False)
                        instances.extend(tmp_instances)
                    
        logging.info("length of instances:{}".format(len(instances)))
        
        if not entailment:
            assert len(instances) == 120*120*sentence_selection_size*2
        else:
            assert len(instances) == 120*120*sentence_selection_size*2*12
        #logging.info("sentence selection dict:{}".format(len(sentence_selection_dict)))
        
        return (instances,self.sentence_selection_dict)
    
    def create_multinstances_sentence_selection_entailment(self, data):
        pass
    def sentence_selection_entailment_group_compair(self,question_data,sentence_selection_size):
        number = int(self.number)
        logging.info(number)
        
        
        question_a_indice = list(question_data[self.wordpair[0]].keys())
        question_a_indice = question_a_indice[sentence_selection_size*number:sentence_selection_size*(number+1)]

        question_b_indice = list(question_data[self.wordpair[1]].keys())
        question_b_indice = question_b_indice[sentence_selection_size*number:sentence_selection_size*(number+1)]
        
        return question_a_indice, question_b_indice
        
    def sentence_selection_entailment_exception(self, question_data, sentence_selection_size):
        """Using this function to fix post-quiz difference for different sentence selection size,
           they should have a least X (sentence selection size) number of same sentences
        """
        instance_dict_10_path = os.path.join(self.processor.output_path, 
                                             'sentence_selection10_instance_dict.pickle')
        question_a_indice = []
        question_b_indice = []
        
        if os.path.exists(instance_dict_10_path):
            instance_dict = pickle_load(self.processor.output_path, 
                                             'sentence_selection10_instance_dict.pickle')
            for guid in instance_dict[self.wordpair_str].keys():
                if len(guid) != 3 and guid[0:2] == '11':
                    question_a_indice.append(guid[2:])
                elif len(guid) != 3 and guid[0:2] == '22':
                    question_b_indice.append(guid[2:])  
                
            while len(question_a_indice) != sentence_selection_size:
                q = random.sample(list(question_data[self.wordpair[0]].keys()),1)
                if q[0] not in question_a_indice:
                    question_a_indice.extend(q)
            while len(question_b_indice) != sentence_selection_size:
                q = random.sample(list(question_data[self.wordpair[1]].keys()),1)
                if q[0] not in question_b_indice:
                    question_b_indice.extend(q)
        else:
            question_a_indice = list(question_data[self.wordpair[0]].keys())
            question_a_indice = question_a_indice[:sentence_selection_size]

            question_b_indice = list(question_data[self.wordpair[1]].keys())
            question_b_indice = question_b_indice[:sentence_selection_size]
        
        return question_a_indice, question_b_indice
    
