from definition import MODEL_ROOT, DATA_ROOT
import numpy as np
import torch
from utils import *
import logging
import sys
import json
import csv
import codecs
import pandas as pd
import random
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from collections import Counter

logging.getLogger().setLevel(logging.INFO)


class SentenceSelection(object):
    def __init__(self, model_path, wordpair,sentence_selection_size,expert,version):

        self.model_path = os.path.join(MODEL_ROOT, model_path)
        logging.info("model_path:{}".format(self.model_path))
        self.wordpair = wordpair
        logging.info(wordpair)
        self.wordpair_str = "|".join(wordpair)
        self.model_index = model_path.split('_')[1]
        self.sentence_selection_size = sentence_selection_size
        self.version = version
        self.expert = expert
        self.feature_info, self.expert_gold_answer = self.load_score()
        self.expert_wordset_gold_answer = self.expert_gold_answer[self.wordpair_str]
        self.expert_word_gold_answer = {
            word: self.expert_wordset_gold_answer[word]
            for word in self.wordpair
        }

    def load_score(self):
        score_file = os.path.join(DATA_ROOT, 'index_to_sentence_id.json')
        with codecs.open(score_file, 'r') as score_infile:
            self.feature_info = json.load(score_infile)
            
        #==== expert 1 MH =====#
        if self.expert == 'mh':
            expert_file = os.path.join(DATA_ROOT, "expert1_annotation_wiki_0422.json")
            with codecs.open(expert_file, 'r', encoding='utf-8') as infile:
                self.expert_gold_answer = json.load(infile)

        #==== expert 2 =====#
        elif self.expert == "expert2":
            expert_file = os.path.join(DATA_ROOT, "expert2_annotation_wiki.json")
            with codecs.open(expert_file, 'r', encoding='utf-8') as infile:
                self.expert_gold_answer = json.load(infile)
        
        elif self.expert == "turkers":
            expert_file = os.path.join(DATA_ROOT, "turkers_annotation_wiki.json")
            with codecs.open(expert_file, 'r', encoding='utf-8') as infile:
                self.expert_gold_answer = json.load(infile)
        
        
        
        elif self.expert == "union":
            expert_file = os.path.join(DATA_ROOT, "expert1_annotation_wiki_0422.json")
            with codecs.open(expert_file, 'r', encoding='utf-8') as infile:
                mh_ans = json.load(infile)
            expert_file = os.path.join(DATA_ROOT, "expert2_annotation_wiki.json")
            with codecs.open(expert_file, 'r', encoding='utf-8') as infile:
                herman_ans = json.load(infile)
            union_ans = {}
            for wordpair_str in mh_ans:
                union_ans[wordpair_str] = {}
                for word in mh_ans[wordpair_str]:
                    union_ans[wordpair_str][word] = list(set(mh_ans[wordpair_str][word] +
                                                       herman_ans[wordpair_str][word]))
                  
            self.expert_gold_answer = union_ans
            
        elif self.expert == "intersection":
            expert_file = os.path.join('data', "expert1_annotation_wiki_0422.json")
            with codecs.open(expert_file, 'r', encoding='utf-8') as infile:
                mh_ans = json.load(infile)
            expert_file = os.path.join('data', "expert2_annotation_wiki.json")
            with codecs.open(expert_file, 'r', encoding='utf-8') as infile:
                herman_ans = json.load(infile)
            intersection_ans = {}
            for wordpair_str in mh_ans:
                intersection_ans[wordpair_str] = {}
                for word in mh_ans[wordpair_str]:
                    intersection_ans[wordpair_str][word] = [
                    sent_id for sent_id in mh_ans[wordpair_str][word] 
                    if sent_id in herman_ans[wordpair_str][word]
                    ]
            self.expert_gold_answer = intersection_ans
            
            
        else:
            logging.warning("unassign answer sheet")
            sys.exit(1)

        return self.feature_info, self.expert_gold_answer

    def convert_expert_answer_to_index(self):
        index_to_sentence_id_g = json_load(DATA_ROOT,'index_to_sentence_id.json')
        expert_gold_answer = self.expert_gold_answer[self.wordpair_str]
        index_to_sentence_id = index_to_sentence_id_g[self.wordpair_str]
        guid_list = []
        
        for word in index_to_sentence_id:
            for iid in index_to_sentence_id[word]:
                if index_to_sentence_id[word][iid][0] in expert_gold_answer[word]:
                    guid_list.append(iid)
                    
        return guid_list
        
    def load_sentence_selection_result(self,file_path):
        """
        can you keep the sentence selection format the same ==?????
        """
        doc = json_load(self.model_path,file_path)
        uniq = doc.copy()
        for word in doc: 
            doc[word] = [i[0] for i in doc[word]]
            doc[word].sort()
            
            for i in doc[word]:
                uniq[word].append(self.feature_info[self.wordpair_str][word][i][0])
            uniq[word]=uniq[word][3:]
            
        return uniq
        '''
        
        if int(key) in [0, 10, 16, 18, 19, 20, 21, 22, 25, 26, 27, 29, 30, 31]:
            logging.info("uniq:{}".format(uniq))
            return uniq
        else:
            return doc
        '''

    def convert_id_to_candidate(self, combination):
        if "'" in combination:
            split_char = "'"
        else:
            split_char = "-"

        data = {
            self.wordpair[0]: [],
            self.wordpair[1]: []
        }

        guid_in_list = []
        for single_id in combination.split(split_char):
            if single_id.isdigit():
                prefix = single_id[:2]
                if prefix == '11' or prefix == '21':
                    data[wordpair[0]].append(single_id)
                else:
                    data[wordpair[1]].append(single_id)
                guid_in_list.append(single_id)

        return data, guid_in_list

    def convert_txt_to_dictionary_id(self, txt_file_path):
        """
        convert txt result to dictionary format, 
        this txt result may contain more than one combination of sentence selection result
        
        """

        with codecs.open(os.path.join(self.model_path, txt_file_path), 'r', encoding='utf-8') as infile:
            doc = infile.readlines()

        best_choice_candidates = []
        lists_guid_in_list = []
        for combination in doc:
            candidate, guid_in_list = self.convert_id_to_candidate(combination)
            best_choice_candidates.append(candidate)
            lists_guid_in_list.append(guid_in_list)

        logging.info("best_choice_candidate {}".format(best_choice_candidates))
        return best_choice_candidates, lists_guid_in_list

    def convert_student_submit_to_dictionary_id(self,target_file_names=None):
        """
        convert student submit to candidates
        """
        if target_file_names is None:
            target_file_names = ['all_student_submit_100_100_v2.pickle', 
                                 'all_student_submit_100_200_v2.pickle',
                                 'all_student_guid_array_100_100_v2.pickle',
                                 'all_student_guid_array_100_200_v2.pickle']
                                 
        all_student_submit = pickle_load(self.model_path, target_file_names[0])
        all_student_submit += pickle_load(self.model_path, target_file_names[1])
        
        all_student_guid_array = pickle_load(self.model_path, target_file_names[2])
        all_student_guid_array += pickle_load(self.model_path, target_file_names[3])
        #print(all_student_guid_array[0:5])
        print(max(all_student_submit))
        
        max_index = np.where(all_student_submit == max(all_student_submit))[0]
        print(max_index)
        best_choice_candidates = []
        lists_guid_in_list = []
        
        for col_index in max_index:
            print(col_index)
            best_example_guid = list(set([guid_list[col_index][:23] for guid_list in all_student_guid_array]))
            #print(best_example_guid)
            assert len(best_example_guid)==1 
            
            candidate, guid_in_list = self.convert_id_to_candidate(best_example_guid[0])
            print("candidate:",candidate)
            best_choice_candidates.append(candidate)
            lists_guid_in_list.append(guid_in_list)

        logging.info("best_choice_candidate {}".format(best_choice_candidates))
        return best_choice_candidates, lists_guid_in_list, max(all_student_submit)

    def convert_student_submit_to_dictionary_id_combination(self, target_file_names=None):
        if target_file_names is None:
            target_file_names = ['all_student_submit_100_100_v2.pickle', 
                                 'all_student_submit_100_200_v2.pickle',
                                 'all_student_guid_array_100_100_v2.pickle',
                                 'all_student_guid_array_100_200_v2.pickle']
                                 
        all_student_submit = pickle_load(self.model_path, target_file_names[0])
        all_student_submit += pickle_load(self.model_path, target_file_names[1])
        
        all_student_guid_array = pickle_load(self.model_path, target_file_names[2])
        all_student_guid_array += pickle_load(self.model_path, target_file_names[3])
        
        k = int(self.sentence_selection_size)
        subset_student_submit = [all_student_submit[i] for i in range(k+1)]
        subset_student_guid_array = [all_student_guid_array[i] for i in range(k+1)]
        
        print(max(subset_student_submit))
        
        max_index = np.where(subset_student_submit == max(subset_student_submit))[0]
        print(max_index)
        best_choice_candidates = []
        lists_guid_in_list = []
        
        for col_index in max_index:
            print(col_index)
            best_example_guid = list(set([guid_list[col_index][:23] for guid_list in subset_student_guid_array]))
            assert len(best_example_guid) == 1
            
            candidate, guid_in_list = self.convert_id_to_candidate(best_example_guid[0])
            print("candidate:",candidate)
            best_choice_candidates.append(candidate)
            lists_guid_in_list.append(guid_in_list)

        logging.info("best_choice_candidate {}".format(best_choice_candidates))
        return best_choice_candidates, lists_guid_in_list,max(all_student_submit)
    
        
    
    
    def convert_to_uniq_id(self, guid_in_list):

        uniq_id_list = []
        uniq_data = {
            self.wordpair[0]: [],
            self.wordpair[1]: []}

        for index in guid_in_list:
            # print(index)
            if index[0:2] == '11':
                word = self.wordpair[0]
            else:
                word = self.wordpair[1]
            uniq_data[word].append(self.feature_info[self.wordpair_str][word][index][0])

        logging.info("uniq index:{}".format(uniq_data))
        return uniq_data
    
    def convert_to_id(self, guid_in_list):

        uniq_id_list = []
        data = {
            self.wordpair[0]: [],
            self.wordpair[1]: []}

        for index in guid_in_list:
            # print(index)
            if index[0:2] == '11':
                word = self.wordpair[0]
            else:
                word = self.wordpair[1]
            data[word].append(index)
        #logging.info("uniq index:{}".format(uniq_data))
        return data

    def load_guid_and_logits(self, target_file_names=None):
        """
        load guid and logits to calculate word choice accuracy
        """
        if target_file_names is None:
            target_file_names = ['all_student_logits_array_15_v2.pickle', 'all_student_guid_array_15_v2.pickle']
        all_student_logits_array = pickle_load(self.model_path, target_file_names[0])

        all_student_guid_array = pickle_load(self.model_path, target_file_names[1])

        return all_student_logits_array, all_student_guid_array
    
    
    def load_full_guid_and_logits(self,full_logits_array=None,full_guid_array=None,start_version=None):
        
        #assert self.sentence_selection_size >= 50:
        
        config_sentence_selection_size = int(self.sentence_selection_size)
        size_to_version = {10:50,20:100,30:150,40:200,50:250,60:300,70:350,80:400,90:450,100:500}
        
        version_dict = {50:0,100:1,150:2,200:3,250:4,300:5,350:6,400:7,450:8,500:9}
        
        this_version = version_dict[size_to_version[int(self.sentence_selection_size)]]
        
        if full_logits_array == None:
            full_logits_array = []
            full_guid_array = []
            start_version = 0
        else: start_version +=1
        self.sentence_selection_size = 50
        logging.info("this version:{}".format(this_version))
        logging.info("start version:{}".format(start_version))
        for i in range(start_version,this_version+1):
            logging.info("version v4_z{}".format(i))
            self.version = "v4_z{}".format(i)
            tmp_student_logits_array, tmp_student_guid_array = self.load_truncated_guid_and_logits()
            full_logits_array += tmp_student_logits_array
            full_guid_array += tmp_student_guid_array
            
        # k = 10, 50
        # k = 20, 100
        # k = 30, 150
        # k =100, 500
        
        self.sentence_selection_size = config_sentence_selection_size
        logging.info("config sentence selection size: {}".format(self.sentence_selection_size))
        
        return full_logits_array,full_guid_array
    
    
        
        
    def load_truncated_guid_and_logits(self, target_file_names=None):
        """
        load guid and logits to calculate word choice accuracy
        """
        ss_size = self.sentence_selection_size

        logging.info("load truncated ss_size:{}".format(ss_size))
        if ss_size == "10":
            size1 = 120
            size2 = 240
        elif ss_size == "30":
            size1 = 360
            size2 = 720
        elif ss_size == "50" or ss_size == 50:
            size1 = 600
            size2 = 1200
            
        if target_file_names is None:
            logging.info('all_student_logits_array_{}_{}_simplified_{}.pickle'.format(ss_size,size1,self.version))
            target_file_names = ['all_student_logits_array_{}_{}_simplified_{}.pickle'.format(ss_size,size1,self.version), 
                                 'all_student_logits_array_{}_{}_simplified_{}.pickle'.format(ss_size,size2,self.version),
                                 'all_student_guid_array_{}_{}_simplified_{}.pickle'.format(ss_size,size1,self.version),
                                 'all_student_guid_array_{}_{}_simplified_{}.pickle'.format(ss_size,size2,self.version)]
                                 
        
        all_student_logits_array = pickle_load(self.model_path, target_file_names[0])
        
        
        all_student_logits_array += pickle_load(self.model_path, target_file_names[1])
        
        all_student_guid_array = pickle_load(self.model_path, target_file_names[2])
        all_student_guid_array += pickle_load(self.model_path, target_file_names[3])
        

        return all_student_logits_array, all_student_guid_array
    

    def align_print_logits_and_array(self, target_file_names):
        all_student_logits_array, all_student_guid_array = self.load_truncated_guid_and_logits(target_file_names)

        for i, j in zip(all_student_logits_array[0:6], all_student_guid_array[0:6]):
            self.print_wordpair_accuracy(i, j)

    @staticmethod
    def create_label_from_guid(single_student_guid_array):
        label_array = []
        count = 0
        for single_guid in single_student_guid_array:

            single_guid = single_guid.split('-')
            count = count + 1
            prefix = single_guid[0][0:2]
            suffix = single_guid[1][0:2]
            if prefix == suffix:
                label = 0  # entail
            else:
                label = 1  # not entail
            label_array.append(label)

        return label_array

    def compute_entailment_accuracy(self):
        all_student_logits_array, all_student_guid_array = self.load_truncated_guid_and_logits()
        all_student_logits_array = torch.tensor([f for f in all_student_logits_array], dtype=torch.long)

        for i, j in zip(all_student_logits_array, all_student_guid_array):
            outputs = np.argmax(i, axis=1)

            labels = self.create_label_from_guid(j)
            labels = torch.tensor([f for f in labels], dtype=torch.long)

            num_correct = np.sum((outputs == labels).numpy())
            # print("accuracy:{}".format(num_correct/len(labels)))
            # break
            
    

    def compute_word_choice_accuracy(self, target_file=None,full_logits_array=None,full_guid_array=None,start_version=None):
        """ for sentence selection only!
        find the best combination in term of word choice accuracy
        return word_choice_accuracy 
        the purpose is to convert entailment accuracy into word choice accuracy
        """
        
        #if int(self.sentence_selection_size) >= 50:
        all_student_logits_array, all_student_guid_array = self.load_full_guid_and_logits(
            full_logits_array ,full_guid_array,start_version)
        #else:
        #    all_student_logits_array, all_student_guid_array = self.load_truncated_guid_and_logits(
        #        target_file)
            
        wordpair_result_array = []
        label_array = []
        accuracy_array = []
        split_chunk_indices = list(range(len(all_student_guid_array)))
        split_chunk_index = [i for i in split_chunk_indices if i % 4 == 0]
        label = []
        for i in range(int(len(split_chunk_indices) / 4)):
            label.append(0)
            label.append(1)
        label_array.append(label)

        for col_index in range(0, 14400):
            wordpair_result = []

            # guid_char = [''.join(chr(i) for i in guid).strip('@') for guid in guid_list]
            for row_index in split_chunk_index:
                current_result = []
                prefix = all_student_guid_array[row_index][col_index].split('-')[0][0:2]
                if prefix == '11' or prefix == '12':
                    if (all_student_logits_array[row_index][col_index][0] >=
                            all_student_logits_array[row_index + 2][col_index][0]):
                        current_result.append(0)
                        wordpair_result.append(0)
                    else:
                        current_result.append(1)
                        wordpair_result.append(1)
                    if (all_student_logits_array[row_index + 1][col_index][1] >=
                            all_student_logits_array[row_index + 3][col_index][1]):
                        current_result.append(1)
                        wordpair_result.append(1)
                    else:
                        current_result.append(1)
                        wordpair_result.append(0)
                elif prefix == '22' or prefix == '21':
                    if (all_student_logits_array[row_index][col_index][1] >=
                            all_student_logits_array[row_index + 2][col_index][1]):
                        current_result.append(0)
                        wordpair_result.append(0)
                    else:
                        current_result.append(1)
                        wordpair_result.append(1)
                    if (all_student_logits_array[row_index + 1][col_index][0] >=
                            all_student_logits_array[row_index + 3][col_index][0]):
                        current_result.append(1)
                        wordpair_result.append(1)
                    else:
                        current_result.append(0)
                        wordpair_result.append(0)
                '''
                if col_index == 0:
                    print("guid:{}\t logits:{}\t".format(
                        all_student_guid_array[row_index][col_index],
                        all_student_logits_array[row_index][col_index],
                        ))
                    print("guid:{}\t logits:{}\t".format(
                        all_student_guid_array[row_index+1][col_index],
                        all_student_logits_array[row_index+1][col_index],
                        ))
                    print("guid:{}\t logits:{}\t".format(
                        all_student_guid_array[row_index+2][col_index],
                        all_student_logits_array[row_index+2][col_index],
                        ))
                    print("guid:{}\t logits:{}\t".format(
                        all_student_guid_array[row_index+3][col_index],
                        all_student_logits_array[row_index+3][col_index],
                        ))
                    print("current_result:{}".format(current_result))
                    sys.exit(1)    
                    '''
            wordpair_result_array.append(wordpair_result)

            accuracy = (np.array(wordpair_result_array[col_index]) == np.array(label_array)).astype(np.float32)

            accuracy_array.append(accuracy)

        return accuracy_array, all_student_logits_array, all_student_guid_array


    def convert_accuracy_array_to_id_score(self, accuracy_array, all_student_guid_array):
        id_score = {}

        #all_student_logits_array, all_student_guid_array = self.load_truncated_guid_and_logits()
        for col_index in range(len(accuracy_array)):
            current_guid = [guid_list[col_index][0:3] for guid_list in all_student_guid_array]

            current_guid = list(set(current_guid))
            current_guid.sort()
            current_guid = "-".join(current_guid)

            current_score = (np.sum(accuracy_array[col_index]) / accuracy_array[col_index].size)
            
            id_score[current_guid] = current_score
            # all_student_score.append(current_score)
            

        output_dict_file = os.path.join(self.model_path, 
                                        "word_choice_id_score_{}_{}_{}.json".format(
                                            self.sentence_selection_size,
                                            self.expert,
                                            self.version))
        output_history(output_dict_file, id_score)
        
        assert len(id_score.keys()) == 14400

        return id_score
    
    def convert_accuracy_array_to_id_score_combination(self,accuracy_array, all_student_guid_array):
        logging.info("combination id_score")
        ss_size = self.sentence_selection_size 
        id_score = {}
        
        
        for col_index in range(len(accuracy_array)):
            if col_index == 0:
                logging.info(accuracy_array[col_index].size)
                logging.info("expect number of questions:{}".format(ss_size*12*5))
            current_guid = [guid_list[col_index][0:3] for guid_list in all_student_guid_array]
            current_guid = list(set(current_guid))
            current_guid.sort()
            current_guid = "-".join(current_guid)
            
            chunks_size = ss_size * 12
            sum_chunks = [np.sum(accuracy_array[col_index][0][x:x+chunks_size]) for x in 
                          range(0,accuracy_array[col_index].size,chunks_size) ]
            
            current_score = [i / (accuracy_array[col_index].size/5) for i in sum_chunks]
            
            id_score[current_guid] = current_score
        
        output_dict_file = os.path.join(self.model_path, 
                                        "word_choice_id_score_combination_{}_{}_{}.json".format(
                                            self.sentence_selection_size,
                                            self.expert,
                                            self.version))
        output_history(output_dict_file, id_score)
        assert len(id_score.keys()) == 14400
        return id_score
    
    
    def separate_id_score_to_types(self,id_score):
        
        type1,type2,type3 = [],[],[]
        expert_answer = self.convert_expert_answer_to_index()
        gold_word_a = [i for i in expert_answer if i[0:2] == '11']
        gold_word_b = [i for i in expert_answer if i[0:2] == '22']

        for key in id_score:
            word_a = key.split('-')[0:3]
            word_b = key.split('-')[3:]
            a_count = 0
            b_count = 0

            for i,j in zip(word_a,word_b):    
                if i in gold_word_a:
                    a_count +=1
                if j in gold_word_b:
                    b_count +=1

            if a_count+b_count >= 4:
                type1.append(key)
            elif a_count + b_count> 1:
                type2.append(key)
            else:
                type3.append(key)
        
        return type1, type2, type3
    
    def compute_quiz_score_correlation_random_sample(self,id_score=None):
        config_sentence_selection_size = int(self.sentence_selection_size)
        size_to_version = {10:50,20:100,30:150,40:200,50:250,60:300,70:350,80:400,90:450,100:500}
        
        version_dict = {50:0,100:1,150:2,200:3,250:4,300:5,350:6,400:7,450:8,500:9}
        
        this_version = version_dict[size_to_version[int(self.sentence_selection_size)]]
        
        output_dict_file = os.path.join("word_choice_id_score_combination_{}_{}_{}.json".format(
                                            self.sentence_selection_size,
                                            self.expert,
                                            "v4_z{}".format(this_version)))
        id_score = json_load(self.model_path,output_dict_file)
        
        type1, type2, type3 = self.separate_id_score_to_types(id_score)
        correlation_by_type = []
        for index,t in enumerate([type1,type2,type3]):
            logging.info("Type {}".format(index))
            type_2_id_score = {key:val[0] for key,val in id_score.items() if key in t}
            Y = list(type_2_id_score.values())
            '''
            X = []
            
            print(Y)
            break
            sys.exit(0)
            for i in range(0,5):
                tmp_x = [score[i] for score in Y]
                X.append(tmp_x)
            '''
            correlation_dict = {
                'quiz1':None,
                'quiz2':None,
                'quiz3':None,
                'quiz4':None,
                'quiz5':None
            }
            
            for j in range(0,20): #random time = 20
                curr_correlation = None
                random_index = random.sample(list(range(len(Y))),100) # random size =100
                curr_Y = [Y[k] for k in random_index]
                
                curr_Y=  pd.DataFrame.from_records(curr_Y ,columns=['quiz1','quiz2','quiz3','quiz4','quiz5'])
                curr_correlation = curr_Y.corr()
                #correlation_list.append(curr_correlation)
                
                if correlation_dict['quiz1'] is None:
                    correlation_dict['quiz1'] = np.array(curr_correlation['quiz1'].to_list())
                    correlation_dict['quiz2'] = np.array(curr_correlation['quiz2'].to_list())
                    correlation_dict['quiz3'] = np.array(curr_correlation['quiz3'].to_list())
                    correlation_dict['quiz4'] = np.array(curr_correlation['quiz4'].to_list())
                    correlation_dict['quiz5'] = np.array(curr_correlation['quiz5'].to_list())
                    
                else:
                    correlation_dict['quiz1'] += np.array(curr_correlation['quiz1'].to_list())
                    correlation_dict['quiz2'] += np.array(curr_correlation['quiz2'].to_list())
                    correlation_dict['quiz3'] += np.array(curr_correlation['quiz3'].to_list())
                    correlation_dict['quiz4'] += np.array(curr_correlation['quiz4'].to_list())
                    correlation_dict['quiz5'] += np.array(curr_correlation['quiz5'].to_list())
                
            
            correlation_dict['quiz1'] = correlation_dict['quiz1']/20
            correlation_dict['quiz2'] = correlation_dict['quiz2']/20
            correlation_dict['quiz3'] = correlation_dict['quiz3']/20
            correlation_dict['quiz4'] = correlation_dict['quiz4']/20
            correlation_dict['quiz5'] = correlation_dict['quiz5']/20
            
            df_dict = pd.DataFrame(correlation_dict)
            #print(df_dict)
            avg = (sum(correlation_dict['quiz1'])+sum(correlation_dict['quiz2'])+sum(correlation_dict['quiz3'])+
                   sum(correlation_dict['quiz4'])+sum(correlation_dict['quiz5']))/25
            correlation_by_type.append(avg)
            
        result = {}
        result[str(self.sentence_selection_size)] = {
                "number_distribution":[len(type1),len(type2),len(type3)],
                "correlation":correlation_by_type
            }
        
        output_dict_file = os.path.join(self.model_path, 
                                        "correlation_of_Ks_{}_v100.json".format(self.expert)
                                       )
        
        
        output_history(output_dict_file, result)
        
        return correlation_by_type

        
    def compute_quiz_score_distance(self,id_score=None):
        config_sentence_selection_size = int(self.sentence_selection_size)
        size_to_version = {10:50,20:100,30:150,40:200,50:250,60:300,70:350,80:400,90:450,100:500}
        
        version_dict = {50:0,100:1,150:2,200:3,250:4,300:5,350:6,400:7,450:8,500:9}
        
        this_version = version_dict[size_to_version[int(self.sentence_selection_size)]]
        
        output_dict_file = os.path.join("word_choice_id_score_combination_{}_{}_{}.json".format(
                                            self.sentence_selection_size,
                                            self.expert,
                                            "v4_z{}".format(this_version)))
        id_score = json_load(self.model_path,output_dict_file)
        
        type1, type2, type3 = self.separate_id_score_to_types(id_score)
        distance_by_type = []
        
        for index,t in enumerate([type1,type2,type3]):
            type_2_id_score = {key:val[0] for key,val in id_score.items() if key in t}
            Y = list(type_2_id_score.values())
            X = []
            for i in range(0,5):
                tmp_x = [score[i] for score in Y]
                X.append(tmp_x)

            matrix3 = pairwise_distances(X, metric='euclidean') 
            distance_by_type.append(np.sum(matrix3)/matrix3.size)
        
        result = {}
        result[str(self.sentence_selection_size)] = {
                "number_distribution":[len(type1),len(type2),len(type3)],
                "distance":distance_by_type
            }
        
        output_dict_file = os.path.join(self.model_path, 
                                        "distance_of_Ks_{}_v2.json".format(self.expert)
                                       )
        
        
        output_history(output_dict_file, result)
        
        return distance_by_type
           
    @staticmethod
    def find_the_best_candidates_from_accuracy_array(self, accuracy_array, all_student_guid_array):
        
        logging.info(len(accuracy_array))
        
        all_student_score = []
        for col_index in range(len(accuracy_array)):
            if col_index == 0:
                print(len(accuracy_array[col_index][0]))
            
            all_student_score.append(np.sum(accuracy_array[col_index]))

        max_index = np.where(all_student_score == max(all_student_score))[0]
        print(max_index)
        best_example_guid_array = []
        for col_index in max_index:
            best_example_guid = [guid_list[col_index][0:3] for guid_list in all_student_guid_array]
            
            best_example_guid_array.append(list(set(best_example_guid)))

        return best_example_guid_array
    
    @staticmethod
    def find_the_single_best_candidates_from_id_score(id_score):
        sum_id_score = {}
        
        index = list(id_score.keys())
        
        for key,vals in id_score.items():
            sum_id_score[key] = sum(vals[0])
             
        curr_scores = [val for key,val in sum_id_score.items()]
        #print(max(curr_scores))
        max_index = [index for index,i in enumerate(curr_scores) if i == max(curr_scores)]
        print(max_index)
        id_list = [index[col_index] for col_index in max_index]
        
        return id_list
    
    @staticmethod
    def find_the_5_group_best_candidates_from_id_score(id_score):
        
        index = list(id_score.keys())
        
        best_example_guid_array = []
        
        candidate_by_group = {}
        for i in range(0,5):
            logging.info("group {}".format(i))
            curr_scores = [val[0][i] for key,val in id_score.items()]
            max_index = [index for index,i in enumerate(curr_scores) if i == max(curr_scores)]
            best_example_guid_array.extend([index[col_index] for col_index in max_index])
            
            #candidate_by_group[i] = {
            #    "id":[index[col_index] for col_index in max_index],
            #    "score":max(curr_scores)
            #}
      
        return  best_example_guid_array
        

    def convert_id_score_to_precision_for_all(self, id_score, id_score_file=None):
        """
        precision_for_all is used for training difficulty score linear regression
        """
        precision_for_all = {}

        if id_score_file is not None:
            with codecs.open(id_score_file, 'r') as infile:
                id_score = json.load(infile)

        for key, value in id_score.items():
            precision_for_all[key] = {'wordpair': {}, 'precision': 0, 'recall': 0}
            print(key, value, self.model_index)
            candidate, guid_in_list = self.convert_id_to_candidate(key)
            if int(self.model_index) >= 10 and int(self.model_index) != 15:
                candidate = self.convert_to_uniq_id(guid_in_list)

            precision, recall = ss.compare_sentence_selection_result_with_expert_choice(candidate)

            precision_for_all[key]['wordpair'] = candidate
            precision_for_all[key]['precision'] = precision
            precision_for_all[key]['recall'] = recall

        # with codecs.open(os.path.join(self.model_path, ' precision_for_all_morethan3.json'),
        #                 'w', encoding='utf-8') as outfile:
        #    json.dump(precision_for_all, outfile, indent=4)

        return precision_for_all
    
    def compare_sentence_selection_result_with_expert_choice_by_word(self, lla_choice):
        correct_num = 0
        ans_type = "precision","recall","accuracy","f1"
        result = {
            self.wordpair[0]:{"precision":{},"recall":{},"accuracy":{}},
            self.wordpair[1]:{"precision":{},"recall":{},"accuracy":{}},
            "total":{"precision":{},"recall":{},"accuracy":{},"f1":{}}
        }
        true_negative = 0
        true_positive = 0
        
        for word in self.wordpair:
            correct_num_by_word = 0
            reference = self.expert_word_gold_answer[word]
            predicted = lla_choice[word]

            repeated = self.expert_wordset_gold_answer["rep_{}".format(word)]
            reference_expand = []
            for r in reference:
                reps = [rep for rep in repeated if r in rep]
                if reps:
                    if len(reps) != 1:
                        print("GGGGGGGGG")
                    reference_expand.append(reps[0])
                else:
                    reference_expand.append([r])
                    
            logging.info("reference expand:{}".format(reference_expand))
            logging.info("predict:{}".format(predicted))
            corrects = {
                [i for i, r in enumerate(reference_expand) if p in r][0]
                for p in predicted if any(p in r for r in reference_expand)
            }
            

            correct_num += len(corrects)
            correct_num_by_word = len(corrects)
            true_positive_by_word = correct_num_by_word
            false_positive_by_word = len(predicted) - correct_num_by_word 
            false_negative_by_word = len(reference) - correct_num_by_word 
            true_negative_by_word = 10 - true_positive_by_word - false_positive_by_word - false_negative_by_word
            
            true_positive += true_positive_by_word
            true_negative += true_negative_by_word
            
            
            precision_by_word = correct_num_by_word /len(lla_choice[word])
            recall_by_word = correct_num_by_word/len(self.expert_word_gold_answer[word])
            
            logging.info("true_positive {}".format(true_positive_by_word))
            logging.info("false_positive {}".format(false_positive_by_word))
            logging.info("false_negative {}".format(false_negative_by_word ))
            logging.info("true_negative {}".format(true_negative_by_word))
            result[word]["precision"] = round(precision_by_word,3)
            result[word]["recall"] = round(recall_by_word,3)
            result[word]["accuracy"] = round((true_positive_by_word+true_negative_by_word)/10,3)
            
            

        precision = correct_num / sum(len(lla_choice[word]) for word in self.wordpair)
        recall = correct_num / sum(len(self.expert_word_gold_answer[word]) for word in self.wordpair)

        if correct_num == 0:
            f = 0
        else:
            f = (2 * precision * recall) / (precision + recall)

        logging.info("precision = {} ".format(round(precision, 3)))
        logging.info("recall = {} ".format(round(recall,3)))
        logging.info("accuracy = {}".format(round((true_positive+true_negative)/20,3)))
        logging.info("f= {}".format(f))
        
        result["total"]["precision"]=round(precision, 3)
        result["total"]["recall"]=round(recall, 3)
        result["total"]["f1"]=round(f, 3)
        result["total"]["accuracy"] = round((true_positive+true_negative)/20,3)
        
        output_dict_file = os.path.join(self.model_path, 
                                        "sentence_selection_analysis_result_{}_{}_{}.json".format(
                                            self.sentence_selection_size,
                                            self.expert,
                                            self.version))
        output_history(output_dict_file, result)
        return precision,recall, result
    
    def compare_sentence_selection_result_with_expert_choice(self, lla_choice):
        correct_num = 0
        for word in self.wordpair:
            reference = self.expert_word_gold_answer[word]
            predicted = lla_choice[word]

            repeated = self.expert_wordset_gold_answer["rep_{}".format(word)]
            reference_expand = []
            for r in reference:
                reps = [rep for rep in repeated if r in rep]
                if reps:
                    if len(reps) != 1:
                        print("GGGGGGGGG")
                    reference_expand.append(reps[0])
                else:
                    reference_expand.append([r])

            corrects = {
                [i for i, r in enumerate(reference_expand) if p in r][0]
                for p in predicted if any(p in r for r in reference_expand)
            }

            correct_num += len(corrects)

        precision = correct_num / sum(len(lla_choice[word]) for word in self.wordpair)
        recall = correct_num / sum(len(self.expert_word_gold_answer[word]) for word in self.wordpair)

        if correct_num == 0:
            f = 0
        else:
            f = (2 * precision * recall) / (precision + recall)

        print("precision = ", round(precision, 3))
        print("recall = ", recall)

        return precision, recall

    def print_instance_file(self, target_file_name):
        info = pickle_load(self.model_path, target_file_name)
        for count, i in enumerate(info):
            print(i.guid)
            if count > 36:
                break

    def load_instance_dict(self):
        set_type = 'test'
        
        self.instance_dict = pickle_load(self.model_path, 'sentence_selection50_instances_v4_z9.pickle'.format(set_type))
        #print(self.instance_dict)

    def convert_guid_to_sentence(self, guid):
        '''
        sentence = []
        for single_guid in guid.split('-'):
            # print(single_guid)
            sentence.append(self.instance_dict[self.wordpair_str][str(single_guid)][0])
        return sentence
        '''
        sents = []
        for single_guid in guid.split('-'):
            if single_guid[0:2] == "11":
                sent = self.feature_info[self.wordpair_str][self.wordpair[0]][single_guid][1]
                sents.append(sent)
            else:
                sent = self.feature_info[self.wordpair_str][self.wordpair[1]][single_guid][1]
                sents.append(sent)
        return sents

    
    def get_sentence_to_id(self):
        with codecs.open('data/wiki_sentences_all_info_parsed_0410.json','r',encoding='utf-8') as infile:
            wiki_info = json.load(infile)

        if self.wordpair_str not in wiki_info:
            print("******",self.wordpair_str)
            return None
        wiki_wordpair_info = wiki_info[self.wordpair_str]

        sentence_20 = {}
        for word in wiki_wordpair_info:
            if word == self.wordpair[0]:
                prefix = "11"
            else:
                prefix == "22"
            sentence_20[word] = {}
            for sentence in wiki_wordpair_info[word]:
                sentence_20[word][sentence["id"]] = sentence["rawSentence"]


            #sentence_20[word] = {k: v for k, v in sorted(sentence_20[word].items(), key=lambda item: item[1])}
            
            
            #sentence_20[word] = {t[0]: (prefix+str(index),t[1]) for index,t in enumerate(sentence_20[word].items())}

        
        return sentence_20
    
    def compare_candidate_with_pool(self,candidate):
        import csv
        filename = "pool_result_{}_{}".format(self.sentence_selection_size,self.version)
        outfile = codecs.open(os.path.join(MODEL_ROOT,filename),'a+',encoding='utf-8')
        outfile_writer = csv.writer(outfile,delimiter='\t')
        #outfile_writer.writerow(['global_index','wordpair','precision','recall'])
        sentence_20 = self.get_sentence_to_id()
        for word in sentence_20:
            for sent_id in sentence_20[word]:
                
                if sent_id not in candidate[word]:
                    outfile_writer.writerow([sentence_20[word][sent_id],"N"])
                else:
                    outfile_writer.writerow([sentence_20[word][sent_id],"Y"])

    def print_wordpair_accuracy(self, logits, guid):
        # set_type = self.set_type
        length = len(logits)
        indices = list(range(length))
        split_index = [i for i in indices if i % 4 == 0]
        wordpair_result = []
        # logging.info(set_type)
        # suffix_label = []
        for index in split_index:
            # TODO: guid prefix, if prefix == 1, check 1,3 otherwise check 2,4
            prefix = guid[index].split('-')[0][0:2]
            suffix = guid[index].split('-')[1][0:2]
            # logging.info(prefix)
            # logging.info(suffix)

            if prefix == '11' or prefix == '12':
                if logits[index][0] > logits[index + 2][0]:
                    wordpair_result.append(0)
                else:
                    wordpair_result.append(1)
                if logits[index + 1][1] > logits[index + 3][1]:
                    wordpair_result.append(1)
                else:
                    wordpair_result.append(0)
            elif prefix == '22' or prefix == '21':
                if logits[index][1] > logits[index + 2][1]:
                    wordpair_result.append(0)
                else:
                    wordpair_result.append(1)
                if logits[index + 1][0] > logits[index + 3][0]:
                    wordpair_result.append(1)
                else:
                    wordpair_result.append(0)

            for single_guid, single_logits in zip(guid[index:index + 4], logits[index:index + 4]):
                sentences = self.convert_guid_to_sentence(single_guid)

                print("guid:{}\t example:{}\t ".format(single_guid.split('-')[0], sentences[0]))
                print("guid:{}\t question:{}\t".format(single_guid.split('-')[1], sentences[1]))

                print("logits:{}\n".format(single_logits))

            current_prediction = wordpair_result[-2:]
            print("w1 group:{} w2 group:{}\n".format(current_prediction[0], current_prediction[1]))

    def print_logits_and_guid(self, logits, guid):

        # guid = [''.join(chr(i) for i in idi).strip('@') for idi in guid]

        length = len(logits)

        indices = list(range(length))
        split_index = [i for i in indices if i % 4 == 0]

        y_label = []
        x1 = []
        x2 = []
        prediction = []
        for index in split_index:
            current_combination_logits = logits[index:index + 4]
            # if index ==0:
            # print(current_combination_logits)
            curr_comb_guid_quest_prefix = [i.split('-')[1][0:2] for i in guid[index:index + 4]]
            w1_index = curr_comb_guid_quest_prefix.index('11')  # x1
            w2_index = curr_comb_guid_quest_prefix.index('22')  # x1_2
            counter_w1_index = curr_comb_guid_quest_prefix.index('12')  # x2
            counter_w2_index = curr_comb_guid_quest_prefix.index('21')  # x2_2

            for i_index in range(index, index + 4):

                current_q_prefix = guid[i_index].split('-')[1][0:2]

                current_example_prefix = guid[i_index].split('-')[0][0:2]
                # logging.info('word choice:{}'.format(current_example_prefix))
                if current_example_prefix == '11':
                    if current_q_prefix == '11':
                        x1.append(current_combination_logits[w1_index][0])
                        x2.append(current_combination_logits[counter_w1_index][0])
                        y_label.append(1)
                        a1 = current_combination_logits[w1_index][0]
                        a2 = current_combination_logits[counter_w1_index][0]

                        if a1 > a2:
                            prediction.append(0)
                        else:
                            prediction.append(1)
                    elif current_q_prefix == '22':
                        x1.append(current_combination_logits[w2_index][1])
                        x2.append(current_combination_logits[counter_w2_index][1])
                        y_label.append(1)
                        a1 = current_combination_logits[w2_index][1]
                        a2 = current_combination_logits[counter_w2_index][1]
                        if a1 > a2:
                            prediction.append(1)
                        else:
                            prediction.append(0)

                    elif current_q_prefix == '12':
                        x1.append(current_combination_logits[counter_w1_index][0])
                        x2.append(current_combination_logits[w1_index][0])
                        y_label.append(-1)

                    elif current_q_prefix == '21':
                        x1.append(current_combination_logits[counter_w2_index][1])
                        x2.append(current_combination_logits[w2_index][1])
                        y_label.append(-1)
                elif current_example_prefix == '22':
                    if current_q_prefix == '11':
                        x1.append(current_combination_logits[w1_index][1])
                        x2.append(current_combination_logits[counter_w1_index][1])
                        y_label.append(1)
                        a1 = current_combination_logits[w1_index][1]
                        a2 = current_combination_logits[counter_w1_index][1]
                        if a1 > a2:
                            prediction.append(0)
                        else:
                            prediction.append(1)
                    elif current_q_prefix == '22':
                        x1.append(current_combination_logits[w2_index][0])
                        x2.append(current_combination_logits[counter_w2_index][0])
                        y_label.append(1)
                        a1 = current_combination_logits[w2_index][0]
                        a2 = current_combination_logits[counter_w2_index][0]
                        if a1 > a2:
                            prediction.append(1)
                        else:
                            prediction.append(0)

                    elif current_q_prefix == '12':
                        x1.append(current_combination_logits[counter_w1_index][1])
                        x2.append(current_combination_logits[w1_index][1])
                        y_label.append(-1)

                    elif current_q_prefix == '21':
                        x1.append(current_combination_logits[counter_w2_index][0])
                        x2.append(current_combination_logits[w2_index][0])
                        y_label.append(-1)

                elif current_example_prefix == '12':
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

                        a1 = current_combination_logits[w1_index][0]
                        a2 = current_combination_logits[counter_w1_index][0]
                        if a2 > a1:
                            prediction.append(1)
                        else:
                            prediction.append(0)

                    elif current_q_prefix == '21':
                        x1.append(current_combination_logits[counter_w2_index][1])
                        x2.append(current_combination_logits[w2_index][1])
                        y_label.append(1)

                        a1 = current_combination_logits[w2_index][1]
                        a2 = current_combination_logits[counter_w2_index][1]
                        if a2 > a1:
                            prediction.append(0)
                        else:
                            prediction.append(1)

                elif current_example_prefix == '21':
                    if current_q_prefix == '11':
                        x1.append(current_combination_logits[w1_index][1])
                        x2.append(current_combination_logits[counter_w1_index][1])
                        y_label.append(-1)

                    elif current_q_prefix == '22':
                        x1.append(current_combination_logits[w2_index][0])
                        x2.append(current_combination_logits[counter_w2_index][0])
                        y_label.append(-1)
                    elif current_q_prefix == '12':
                        x1.append(current_combination_logits[counter_w1_index][1])
                        x2.append(current_combination_logits[w1_index][1])
                        y_label.append(1)

                        a1 = current_combination_logits[w2_index][1]
                        a2 = current_combination_logits[counter_w2_index][1]
                        if a2 > a1:
                            prediction.append(1)
                        else:
                            prediction.append(0)

                    elif current_q_prefix == '21':
                        x1.append(current_combination_logits[counter_w2_index][0])
                        x2.append(current_combination_logits[w2_index][0])
                        y_label.append(1)
                        a1 = current_combination_logits[w2_index][0]
                        a2 = current_combination_logits[counter_w2_index][0]
                        if a2 > a1:
                            prediction.append(0)
                        else:
                            prediction.append(1)
            for single_guid, single_logits in zip(guid[index:index + 4], current_combination_logits):
                print("guid:{}\t logits:{}\t".format(single_guid, single_logits))

            current_prediction = prediction[-2:]
            print("w1 group:{} w2 group:{}".format(current_prediction[0], current_prediction[1]))


def find_weight_given_precision_and_difficulty(self, output_list, precision_for_all_list):
    ss = []
    ds = []
    precision = []
    for output_dir, precision_for_all in zip(output_list, precision_for_all_list):
        with codecs.open(os.path.join(MODEL_ROOT, output_dir, 'id_diff_score.json'), 'r',
                         encoding='utf-8') as infile:
            id_diff_score = json.load(infile)
        tmp_ss = [id_diff_score[single_id][0] for single_id in id_diff_score]  # x1
        tmp_ds = [id_diff_score[single_id][1] for single_id in id_diff_score]  # x2
        # ds = np.array(ds)*-1
        tmp_precision = [precision_for_all[single_id]['precision'] for single_id in precision_for_all]

        ss = ss + tmp_ss
        ds = ds + tmp_ds
        precision = precision + tmp_precision
    X = [precision, ss, ds]
    df = pd.DataFrame(X)
    df = df.transpose()
    df.columns = ["y", "x1", "x2"]
    X = df.iloc[:, df.columns != 'y']
    Y = df.iloc[:, 0]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)
    coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    return coeff_df
    
class SelectionAnalysis(object):
    
    def __init__(self, output_dir, ss, wordpair,key, sentence_selection_size, expert,version,mode):
        self.ss = ss
        self.key = key
        self.wordpair = wordpair
        self.output_dir = output_dir
        self.sentence_selection_size = sentence_selection_size
        self.expert = expert
        self.version = version
        self.mode = mode
        
        self.all_candidate = None
        self.curr_best = None
        
    def convert_id_to_candidate(self,guid_list):
        if self.expert == "mh":
            #if int(key) in [0, 10, 16, 18, 19, 20, 21, 22, 25, 26, 27, 29, 30, 31]:    
             candidate = self.ss.convert_to_uniq_id(guid_list)
             #else:
             #   candidate = self.ss.convert_to_id(guid_list)
        elif self.expert == "expert2" or self.expert == "turkers":
            candidate = self.ss.convert_to_uniq_id(guid_list)
            
        self.update_all_candidate(candidate)
        
        return candidate
   
    def update_all_candidate(self,candidate):
        if self.all_candidate == None:
            self.all_candidate = candidate      
        else:
            self.all_candidate[self.wordpair[0]].extend(candidate[self.wordpair[0]])
            self.all_candidate[self.wordpair[1]].extend(candidate[self.wordpair[1]])
            
    def custom_filename(self,analysis_type):
        filename = '0601_ss_result_{}_simplified_{}_{}_{}_{}'.format(
            analysis_type,
            self.mode, # nc or not
            self.sentence_selection_size,
            self.expert, self.version)
        
        return filename

    def write_result_to_csv_file(self,key,output_dir, filename, info):
        import csv
        
        outfile = codecs.open(os.path.join(MODEL_ROOT,filename),'a+',encoding='utf-8')
        logging.info("write:{}".format(os.path.join(MODEL_ROOT,filename)))
        outfile_writer = csv.writer(outfile,delimiter=',')
       
        for wordpair in info:
            outfile_writer.writerow(
                (key,
                 wordpair,
                 info[wordpair][0],
                 info[wordpair][1],
                 info[wordpair][2],
                 info[wordpair][3]))
        outfile.close()   

    def most_common(self):
        logging.info("MOST COMMON ANALYSIS")
        all_candidate_most_frequent = {}
        result = {}
        for word in self.all_candidate:
            c = Counter(self.all_candidate[word]).most_common(3)
            all_candidate_most_frequent[word]= [i[0] for i in c]
        #logging.info("all_candidate_most_fequent:{}".format(all_candidate_most_frequent))
        
        precision, recall, _ = self.ss.compare_sentence_selection_result_with_expert_choice_by_word(all_candidate_most_frequent)
        result[",".join(self.wordpair)] = [
            precision,  recall,
            _['total']['accuracy'],
            "|".join(all_candidate_most_frequent[wordpair[0]]+all_candidate_most_frequent[wordpair[1]])
            ]
        
        self.write_result_to_csv_file(
            self.key,self.output_dir,
            self.custom_filename('most_common'),
            result)  
    
    def all_choice(self):
        logging.info("ALL CHOICE ANALYSIS")
        all_candidate = {}
        result = {}
        for word in self.all_candidate:
            all_candidate[word] = list(set(self.all_candidate[word]))
        
        precision, recall, _ = self.ss.compare_sentence_selection_result_with_expert_choice_by_word(all_candidate)
        result[",".join(self.wordpair)] = [
            precision,
            recall,
            _['total']['accuracy'],
            (len(all_candidate[wordpair[0]]),len(all_candidate[wordpair[1]]))
            ]
        
        self.write_result_to_csv_file(
            key,output_dir,
            self.custom_filename('all_set'),
            result)    
        
    def analysis(self,guid_list,id_score=None,score=None,k=None):
        candidate = self.convert_id_to_candidate(guid_list)
        precision, recall = self.ss.compare_sentence_selection_result_with_expert_choice(candidate)
        
        guid_list.sort()
        result = {}
        if self.curr_best == None:
            self.curr_best = (precision, recall,"-".join(guid_list))
        elif precision > curr_best[0]:
            self.curr_best = (precision, recall,"-".join(guid_list))
    
        result[",".join(wordpair)] = [
            precision, recall,
            "-".join(guid_list), 
            id_score["-".join(guid_list)][0] if id_score != "" else score]
        # if k!= None
        # id_score["-".join(guid_list)][0][k] if id_score != "" else score 
        
        self.write_result_to_csv_file(
            self.key, self.output_dir,
            self.custom_filename('full'),
            result)
        
        return self.curr_best
    def find_best_r(self):
        version_dict = {"10":0,"20":1,"30":2,"40":3,"50":4,"60":5,"70":6,"80":7,"90":8,"100":9}
        
        correlation_dict = json_load(os.path.join(MODEL_ROOT,self.output_dir),'correlation_of_Ks_mh_v100.json')
        standard = 0.7
        curr_best = (0,None)
        for key, val in correlation_dict.items():
           # print(val[0]['correlation'])
            
            curr_correlation = val[0]['correlation'][0]
            #print(curr_correlation)
            if curr_correlation > standard:
                print(key)
                return key,version_dict[key]
            elif curr_correlation > curr_best[0]:
                curr_best = (curr_correlation,key)
        
        print(curr_best[1])
        return curr_best[1],version_dict[curr_best[1]]
                                         
            
if __name__ == '__main__':
    wordpair = sys.argv[1:3]
    output_dir = sys.argv[3]
    key = output_dir.split('_')[1]
    setting = sys.argv[4]
    mode = sys.argv[5]
    sentence_selection_size = sys.argv[6]
    expert = sys.argv[7]
    version = sys.argv[8]
    # type of sentence selection: FITB, Entailment, WordChoice
    ss = SentenceSelection(output_dir, wordpair, sentence_selection_size,expert,version)
    sa = SelectionAnalysis(output_dir, ss, wordpair, key, sentence_selection_size, expert,version,mode)
    
    if setting == "FITB" or setting == "fitb":
        
        result = {}
        
        best_choice_candidates, lists_guid_in_list,score = ss.convert_student_submit_to_dictionary_id()
        
        for candidate, guid_list in zip(best_choice_candidates, lists_guid_in_list):
            curr_best = sa.analysis(guid_list, id_score="", score=score)

        result[",".join(wordpair)] = [
            curr_best[0], 
            curr_best[1],
            curr_best[2],
            score]
        
        sa.write_result_to_csv_file(
            key, output_dir,
            sa.custom_filename('best'),
            result
        )
        sa.most_common()
        sa.all_choice()

    if setting == "baseline":
        result = {}
        logging.info("baseline result")
        result_path = "bert_ss_result.json"
        candidate = ss.load_sentence_selection_result(result_path)
        precision, recall = ss.compare_sentence_selection_result_with_expert_choice(candidate)
        result[",".join(wordpair)] = [precision, recall]
        
        sa.write_result_to_csv_file(key, output_dir, result)
    
    if setting == 'regular':
        txt_file_path = 'entailment_sentence_selection_15_v2.txt'

        best_choice_candidates, lists_guid_in_list = ss.convert_txt_to_dictionary_id(txt_file_path)

        for candidate, guid_list in zip(best_choice_candidates, lists_guid_in_list):
            if int(key) in [0, 10, 16, 18, 19, 20, 21, 22, 25, 26, 27, 29, 30, 31]:
                candidate = ss.convert_to_uniq_id(guid_list)
            precision, recall = ss.compare_sentence_selection_result_with_expert_choice(candidate)
            
    if setting == "grid_search":
        logging.info("grid search")
        
        accuracy_array, all_student_logits_array, all_student_guid_array = ss.compute_word_choice_accuracy()
        id_score = ss.convert_accuracy_array_to_id_score_combination(accuracy_array, all_student_guid_array)
        #id_score = json_load('model_entailment_0304/model_0_entailment','word_choice_id_score_combination_100_mh_v4_z9.json')
        dis = ss.compute_quiz_score_distance(id_score)
        ss.compute_quiz_score_correlation_random_sample(id_score)
        
    
    if setting == "grid_search_draft":
        previous_version = 0
        all_student_guid_array = None
        
        for index,i in enumerate(range(10,110,10)):
            if i < int(sentence_selection_size):
                continue
            logging.info("grid search")
            if all_student_guid_array is None:
                ss.version = index
                ss.entence_selection_size = i
                accuracy_array, all_student_logits_array, all_student_guid_array = ss.compute_word_choice_accuracy()
                previous_version = index
            else:
                logging.info("this version:{} sentense selection size:{}".format(index,i))
                ss.version = index
                ss.sentence_selection_size = i
                accuracy_array, all_student_logits_array, all_student_guid_array = ss.compute_word_choice_accuracy(
                    full_logits_array = all_student_logits_array,
                    full_guid_array = all_student_guid_array,
                    start_version=previous_version
                )
                previous_version = index


            id_score = ss.convert_accuracy_array_to_id_score_combination(accuracy_array, all_student_guid_array)
            #id_score = json_load('model_entailment_0304/model_0_entailment','word_choice_id_score_combination_100_mh_v4_z9.json')
            #dis = ss.compute_quiz_score_distance(id_score)
            #ss.compute_quiz_score_correlation_random_sample(id_score)
            del id_score
            
    if setting == "best_r":
        result = {}
        k,k_version = sa.find_best_r()
        output_dict_file = os.path.join(
            output_dir,
            "word_choice_id_score_combination_{}_{}_v4_z{}.json".format(
                k,expert,k_version))
        
        id_score = json_load(MODEL_ROOT,output_dict_file)
        logging.info("start find the single best")
        id_list = ss.find_the_single_best_candidates_from_id_score(id_score=id_score)
        
        for guid_list in id_list:
            
            curr_best = sa.analysis(guid_list.split('-'),id_score=id_score,score="",k=None)
            
        result["|".join(wordpair)]= [
            curr_best[0], 
            curr_best[1],
            curr_best[2],
            id_score[guid_list][0]
        ]
            
        output_dict_file = os.path.join(
            MODEL_ROOT,
            output_dir, 
            "sentence_selection_by_group_single_best{}_{}.json".format(expert,k_version))
        output_history(output_dict_file, result)
        
    if setting == "word_choice_by_k":
        output_dict_file = os.path.join(
            output_dir,
            "word_choice_id_score_combination_{}_{}_{}.json".format(
                sentence_selection_size,"mh",version))
        
        id_score = json_load(MODEL_ROOT,output_dict_file)
        #candidate_id_by_group = ss.find_the_single_best_candidates_from_id_score(id_score=id_score)
        id_list = ss.find_the_5_group_best_candidates_from_id_score(id_score=id_score)
        
        result= {}

        for guid_list in id_list: #candidate_id_by_group[group]['id']:
            curr_best = sa.analysis(guid_list.split('-'),id_score=id_score,score="",k=None)

        result["|".join(wordpair)]=[
        curr_best[0], 
        curr_best[1],
        curr_best[2],
        id_score[guid_list][0]]

        sa.write_result_to_csv_file(
            key,output_dir,
            sa.custom_filename('best'),
            result
        )
        sa.most_common()
        sa.all_choice()
                
    if setting == 'word_choice':  # then find the best candidate using logits
        logging.info("word_choice")
        #accuracy_array, all_student_guid_array = ss.compute_word_choice_accuracy()
        #id_score = ss.convert_accuracy_array_to_id_score(accuracy_array, all_student_guid_array)
        
        output_dict_file = os.path.join(
            output_dir,
            "word_choice_id_score_combination_{}_{}_{}.json".format(
                sentence_selection_size,expert,version))
        
        id_score = json_load(MODEL_ROOT,output_dict_file)
        
        best_choice_candidates = ss.find_the_best_candidates_from_accuracy_array(
            accuracy_array,  all_student_guid_array)
        logging.info(len(best_choice_candidates))
        result = {}
        
        for guid_list in best_choice_candidates:
            curr_best = sa.analysis(guid_list,id_score=id_score,score="",k=None)
            
        result[",".join(wordpair)] = [
            curr_best[0], 
            curr_best[1],
            curr_best[2],
            id_score["-".join(guid_list)]]
        
        sa.write_result_to_csv_file(
            key,output_dir,
            sa.custom_filename('best'),
            result
        )
        sa.most_common()
        sa.all_choice()
        

    # re-train difficulty weight when model's logits changes
    if setting == 're-train':
        output_list = ['model_15_entailment_entail_clozeQ', 'model_27_entailment_entail_clozeQ']
        wordpair_list = ['particular', 'peculiar'], ['accountability', 'liability']
        precision_list = []
        for output_dir, wordpair in zip(output_list, wordpair_list):
            ss = SentenceSelection(output_dir, wordpair)
            filenames = ['test_logits_array_v2_5e-05.pickle', 'test_guid_array_v2_5e-05.pickle']
            accuracy_array = ss.compute_word_choice_accuracy(filenames)
            id_score = ss.convert_accuracy_array_to_id_score(accuracy_array)
            precision_for_all = ss.convert_id_score_to_precision_for_all(id_score)
            precision_list.append(precision_for_all)

        coeff = find_weight_given_precision_and_difficulty(output_list, precision_list)
        logging.info("New Coefficient:{}".format(coeff))

    #
    # ss.print_instance_file('test_instances.pickle')
    # ss.compute_entailment_accuracy()
    # ss.convert_guid_to_sentence()
    if setting =="GiveMeExample":
        #result_path = os.path.join("GiveMeExample_Result",wordpair_str+"_result_id.json")
        doc= codecs.open('GiveMeExample_precison_0601_{}.tsv'.format(expert),'a+',encoding='utf-8')
        csv_writer = csv.writer(doc, delimiter='\t')
        candidate = json_load("GiveMeExample_Result","|".join(wordpair)+"_result_id.json")
        precision, recall = ss.compare_sentence_selection_result_with_expert_choice(candidate)
        print("|".join(wordpair),precision,recall)
        csv_writer.writerow(["|".join(wordpair),precision,recall])
    if setting == "other":
        doc_random= codecs.open('random_precison.tsv','w+',encoding='utf-8')
        csv_writer = csv.writer(doc_random, delimiter='\t')
        random_choice = json_load(".",'random.json')
        for wordpair_str in random_choice:
            ss = SentenceSelection(output_dir, wordpair_str.split('|'), sentence_selection_size,expert,version)
                
            candidate = ss.convert_to_id(random_choice[wordpair_str])
            precision, recall = ss.compare_sentence_selection_result_with_expert_choice(candidate)
            print(wordpair_str,precision,recall)
        
            csv_writer.writerow([wordpair_str,precision,recall])