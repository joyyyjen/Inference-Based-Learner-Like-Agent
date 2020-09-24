from torch.utils.data import (DataLoader, RandomSampler,
                              SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from pytorch_pretrained_bert_modified.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert_modified.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert_modified.tokenization import BertTokenizer
from pytorch_pretrained_bert_modified.optimization import BertAdam, warmup_linear
from Indexer import FindIndex
from DataProcess_v2 import DataProcess
from Features import *
from utils import pickle_load, pickle_dump, output_history
from definition import MODEL_ROOT
import numpy as np
import torch
import random
import time
import os
import sys
import math
import codecs
import logging
logging.getLogger().setLevel(logging.INFO)


class EntailmentModel(object):
    def __init__(self, args, device,version):
        """ Initialize and update arguments including define wordpair definitions
            for example, 
                wordpair: ['responsibility','liability','elderly','senior']
                wordpair_list: [['responsibility', 'liability'], ['elderly', 'senior']] 
                wordpair_name: ['responsibility|liability', 'elderly|senior'] 
                wordpair_dict: {'responsibility|liability': ['responsibility', 'liability'], 
                                  'elderly|senior': ['elderly', 'senior']}            
        """
        logging.info("ENTAILMENT MODEL")
        self.device = device
        self.version = version
        self.args = args
        
        self.update_args()
        self.output_path = os.path.join(MODEL_ROOT, args.output_dir)
        self.indexer = FindIndex(self.output_path)
        self.processor = DataProcess(self.output_path, self.indexer)
        
        self.args.train_batch_size = self.args.train_batch_size // args.gradient_accumulation_steps
        self.label_list = ['True', 'False']
        self.num_labels = len(self.label_list)
        
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_1')
        
        self.wordpair = self.args.wordpair.split(",")
        self.wordpair_list = [self.wordpair[2*i:2*(i+1)] for i in range(int(len(self.wordpair)/2))]
        self.wordpair_name = ["|".join(pair) for pair in self.wordpair_list ] 
        self.wordpair_dict = {
            name: pair for name, pair in zip(self.wordpair_name, self.wordpair_list)
             }

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
        self.transform_mapping = {}
        self.optmizer = None
        self.model = None
        
    def update_args(self):
        """update arguments for entailment model only"""
        
        if len(self.device) ==3: 
            self.args.train_batch_size = 36
        self.args.eval_batch_size = self.args.eval_batch_size*6
        self.args.max_seq_length = 128
        self.args.version = self.version
        
        if self.args.do_word_choice_train:
            self.args.num_train_epochs = 1 
    
    def generate_data(self):
        """generate mapping for near-synonym interchangeability
        generate data for model usage
        """
        
        logging.info("GENERATE MAPPING")
        
        for pair in self.wordpair_dict:
            self.transform_mapping[pair] = self.processor.generate_mapping(self.wordpair_dict[pair])
            logging.info("GENERATE DATA")
            self.processor.generate_data(self.wordpair_dict[pair],self.transform_mapping[pair],self.args.uniq)
        
    def get_instances(self,set_type,setting=None):
        """
        get multiprocessor instances and save instance in dictionary format to preprocess tokenization
        """
        
        logging.info('set_type:{} \t setting:{}'.format(set_type,setting))
        instances, instance_dict = self.processor.get_multipair_instance(
            self.wordpair_list, self.args, set_type, setting)
        
        if setting is not None:
            set_type = set_type + '_' + setting
            
        if set_type == 'sentence_selection':
            set_type = set_type + str(self.args.sentence_selection_size)
            
        pickle_dump(self.output_path, '{}_instances_{}.pickle'.format(set_type,self.version), instances)
        instance_dict = dict_tokenizer(self.tokenizer, self.wordpair_dict,instance_dict, self.indexer)
        pickle_dump(self.output_path, '{}_instance_dict_{}.pickle'.format(set_type,self.version), instance_dict)
        
        logging.info("  Num instances = %d", len(instances))
        
        return instances, instance_dict
    
    def load_instances_dict(self, set_type, setting=None):
        
        if setting is not None:
             set_type = set_type + '_' + setting 
        if set_type == 'sentence_selection':
            set_type = set_type + str(self.args.sentence_selection_size)
        
        
        instance_dict = pickle_load(self.output_path,'{}_instance_dict_{}.pickle'.format(set_type,self.version))
        
        return instance_dict
    
    def initialize_optimizer(self, train_instances_length):
        
        num_train_optimization_steps = int(
            train_instances_length / self.args.train_batch_size / self.args.gradient_accumulation_steps
            ) * self.args.num_train_epochs

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                 {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                 {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                 ]

        self.optimizer = BertAdam(
            optimizer_grouped_parameters, lr=self.args.learning_rate,
            warmup=self.args.warmup_proportion, t_total=num_train_optimization_steps)
    
        logging.info("  Batch size = %d", self.args.train_batch_size)
        logging.info("  Num steps = %d", num_train_optimization_steps)
        
    def convert_instance_to_uniq_features(self, transpose, instance_dict, set_type, setting=None):
        
        example_question_set = {}
        for i in transpose:
            if i.guid not in example_question_set:
                example_question_set[i.guid] = [1,i]
            else:
                example_question_set[i.guid][0] += 1
        example_question_list = [val[1] for key,val in example_question_set.items()]

        features = convert_instances_to_features_entailment(example_question_list,
                self.label_list, self.args.max_seq_length, self.tokenizer, self.indexer,
                instance_dict,set_type, self.args.dropout)
        
        if setting is not None:
             set_type = set_type + '_' + setting 
        pickle_dump(self.output_path, '{}_{}_uniq_features_{}.pickle'.format(
            set_type,self.args.sentence_selection_size,self.version), features)
        logging.info("  Num Features = %d", len(features))
        return features

    def convert_instance_to_features(self, instances, instance_dict, set_type, setting=None):
        
        features = convert_instances_to_features_entailment(instances,
                self.label_list, self.args.max_seq_length, self.tokenizer, self.indexer,
                instance_dict,set_type, self.args.dropout)
        if setting is not None:
             set_type = set_type + '_' + setting 
        pickle_dump(self.output_path, '{}_features_{}.pickle'.format(set_type,self.version), features)
        logging.info("  Num Features = %d", len(features))
        return features
    
    def load_features(self, set_type, setting=None):
        
        if setting is not None:
             set_type = set_type + '_' + setting 
        features = pickle_load(self.output_path, '{}_features_{}.pickle'.format(set_type,self.version))
        return features
    
    def transpose_sentence_selection_instance(self, test_instances):
        complete_group_size = 24 * self.args.sentence_selection_size
        chunks = [test_instances[x:x+complete_group_size] 
                  for x in range(0, len(test_instances), complete_group_size)]
        question_level_dictionary = {}
        for question_level in range(complete_group_size):
            
            for id_num in range(len(chunks)):
                
                if question_level not in question_level_dictionary:
                    question_level_dictionary[question_level] = []
                question_level_dictionary[question_level].append(chunks[id_num][question_level])
        
        transpose = []
        for q in question_level_dictionary:
            transpose.extend(question_level_dictionary[q])
        
        pickle_dump(self.output_path, 'sentence_selection_transpose_{}.pickle'.format(self.version), transpose)
        return transpose

    def train(self):
        
        self.model = BertForSequenceClassification.from_pretrained(
            self.args.bert_model, cache_dir=self.cache_dir,
            num_labels=self.num_labels, similarity_mode=self.args.similarity)
        self.model.to(self.device[0])  
        self.model = torch.nn.DataParallel(self.model, device_ids=self.args.device_ids)
        
        if self.args.new_mode:
            
            # self.generate_data() # old data
            train_instances, instance_dict = self.get_instances(set_type='train', setting=None)
            
            train_features = self.convert_instance_to_features(train_instances, instance_dict, 'train')
            self.initialize_optimizer(len(train_instances))
        else:
            instance_dict = self.load_instances_dict(set_type='train', setting=None)
            train_features = self.load_features(set_type='train', setting=None)
            self.initialize_optimizer(len(train_features))
            
        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0 
        
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        
        len_train_features = len(train_features)
        train_features = None

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.args.train_batch_size)

        self.model.train()
        start_time = time.time()
        
        save_times = 0
        for _ in trange(int(self.args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_instances, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device[0]) for t in batch)

                input_ids, input_mask, segment_ids, label_ids= batch
                loss, logits, embedding = self.model(input_ids, segment_ids,input_mask, label_ids)

                if len(self.device) > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_instances += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                if step % math.floor(len_train_features/self.args.train_batch_size/2) ==0:
                #if step % 1000 == 0 :
                    save_times += 1
                    logging.info("\rEpoch {} batch= {:>4} loss = {:.5f}".format(_, step, tr_loss/nb_tr_steps, end="\r"))
                    self.save_result(epoch=_, step=step, loss=(tr_loss/nb_tr_steps), mode='model_result')
                    self.save_whole_model(epoch=_, loss=(tr_loss/nb_tr_steps),global_step=global_step)
                    self.evaluation(set_type='dev', step=step)

        end_time = time.time()
        #self.save_result(mode = 'model_time',start_time=start_time,end_time=end_time,)
        logging.info('save model state dict')
        self.save_model_state_dict(setting='train')

    def word_choice_train(self):
        input_model_file = os.path.join(self.output_path, WEIGHTS_NAME)
        self.model = BertForSequenceClassification(self.args.bert_model,
                                                   num_labels=num_labels,
                                                   similarity_mode=self.args.similarity)
        
        logging.info("model to tune:{}".format(input_model_file))
        self.model.load_state_dict(torch.load(input_model_file, map_location=self.device[0]))
        
        self.model = torch.nn.DataParallel(self.model, device_ids=self.args.device_ids)
        
        if self.args.new_mode:
            
            train_instances, instance_dict = self.get_instances(set_type='train', setting='word_choice')

            train_features = self.convert_instance_to_features(train_instances, instance_dict, 'train')
            self.initialize_optimizer(len(train_instances))
        else:
            instance_dict = self.load_instances_dict(set_type='train', setting='word_choice')
            train_features = self.load_features(set_type='train', setting='word_choice')
            self.initialize_optimizer(len(train_features))
            
        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0  
        
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_guid = torch.tensor([f.guid for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_guid)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size= self.args.train_batch_size)

        self.model.train()
        # ---- parameter to keep track on ----#
        #loss_array = []
        logits_array = []
        guid_array = []
        embedding_array = []
    
        for _ in trange(int(self.args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_instances, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device[0]) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, guid = batch
                loss, logits, last_layer_embedding = self.model(input_ids, segment_ids, input_mask, label_ids, guid)
                
                # ---- parameters to keep track on ----#
                guid = list(guid.detach().cpu().numpy())
                guid_array.append(
                    [''.join(chr(i) for i in single_id).strip('@') for single_id in guid]
                )
                
                logits_copy = logits.clone()
                logits_copy = logits_copy.detach().cpu().numpy()
                logits_array.append(logits_copy)
                
                last_layer_embedding_copy = last_layer_embedding.clone()
                last_layer_embedding_copy = last_layer_embedding_copy.detach().cpu().numpy()
                embedding_array.append(last_layer_embedding_copy)

                if len(self.device) > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_instances += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()

                    self.optimizer.zero_grad()

                #if step % 1000 == 0:
                if step % math.floor(len(train_features)/self.args.train_batch_size/4) ==0:
                    epoch = _
                    curr_loss = tr_loss/nb_tr_steps
                    logging.info("\rEpoch {} batch= {:>4} loss = {:.5f}".format(
                        epoch=epoch,
                        step=step,
                        loss=curr_loss,
                        end="\r"))
                    self.save_result(epoch=epoch, step=step, loss=loss, mode='model_result')

        logging.info('start saving model')
        self.save_whole_model(epoch=_, loss=(tr_loss/nb_tr_steps), global_step=global_step)
        self.save_model_state_dict(setting='word_choice')
        # save_result(logits_array = logits_array,guid_array = guid_array,embedding_array = embedding_array)

    def evaluation(self, set_type, step):
        """
        validation function
        """
        
        if os.path.exists(os.path.join(self.output_path, '{}_features.pickle'.format(set_type))):
            # instance_dict = self.load_instances_dict(set_type='dev', setting=None)
            eval_features = self.load_features(set_type='dev', setting=None)
              
        else:
            eval_instances, instance_dict = self.get_instances(set_type='dev', setting=None)
            eval_features = self.convert_instance_to_features(eval_instances, instance_dict, 'dev')
            
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_guid = torch.tensor([f.guid for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids,all_input_mask, all_segment_ids, all_label_ids, all_guid)
        eval_sampler = SequentialSampler(eval_data)
        batch_size = 36
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        self.model.eval()

        eval_loss, eval_accuracy, word_eval_accuracy = 0, 0, 0
        nb_eval_steps, nb_eval_instances = 0, 0
        logits_array = []
        guid_array = []
        embedding_array = []
        
        for input_ids, input_mask, segment_ids, label_ids, guid in tqdm(eval_dataloader,desc = "Evaluation"):
            input_ids = input_ids.to(self.device[0])
            input_mask = input_mask.to(self.device[0])
            segment_ids = segment_ids.to(self.device[0])
            label_ids = label_ids.to(self.device[0])

            with torch.no_grad():
                tmp_eval_loss, logits, embedding = self.model(input_ids, segment_ids, input_mask, label_ids)
            
            # ---- parameters to keep track on ----#
            
            logits = logits.detach().cpu().numpy()
            logits_array.append(logits)
            embedding = embedding.detach().cpu().numpy()
            embedding_array.append(embedding)
            label_ids = label_ids.to('cpu').numpy()

            tmp_eval_accuracy = accuracy(logits,label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_instances += input_ids.size(0)
            nb_eval_steps += 1

            guid_char = [''.join(chr(i) for i in idi).strip('@') for idi in guid]
            guid_array.append(guid_char)
            tmp_wordpair_accuracy = (wordpair_accuracy(logits,guid_char,'normal')/ int(len(logits)/2))
            
            word_eval_accuracy += tmp_wordpair_accuracy

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_instances
        word_eval_accuracy = word_eval_accuracy/  nb_eval_steps

        result = {
            'batch': step,
            'eval_loss': eval_loss,
            'eval_accuracy': eval_accuracy,
        }
        logging.info("\r batch= {:>4} eval loss = {:.5f} eval accuracy={:.5f}"
                     .format(result['batch'], result['eval_loss'], result['eval_accuracy'], end="\r"))
        
        output_eval_file = os.path.join(self.output_path, "dev_eval_result")
        output_history(output_eval_file, result)
       
    def test(self,version=None):
        """
        testing function
        """
        if version: 
            self.load_whole_model(version)
        
        else:
            self.load_model_default(setting=None)
        
        
        if self.args.new_mode:
            eval_instances, instance_dict = self.get_instances(set_type='test', setting=None)
            eval_features = self.convert_instance_to_features(eval_instances, instance_dict, 'test')
        else:
            instance_dict = self.load_instances_dict(set_type='test', setting=None)
            eval_features = self.load_features(set_type='test', setting=None)
            
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_guid = torch.tensor([f.guid for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_guid)
        eval_sampler = SequentialSampler(eval_data)
        batch_size = 36
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size =batch_size)

        self.model.eval()
        
        eval_loss, eval_accuracy, word_eval_accuracy = 0, 0, 0
        nb_eval_steps, nb_eval_instances = 0, 0
        logits_array = []
        guid_array = []
        embedding_array = []
        
        for input_ids, input_mask, segment_ids, label_ids, guid in tqdm(eval_dataloader,desc="Evaluation"):
            input_ids = input_ids.to(self.device[0])
            input_mask = input_mask.to(self.device[0])
            segment_ids = segment_ids.to(self.device[0])
            label_ids = label_ids.to(self.device[0])

            with torch.no_grad():
                tmp_eval_loss, logits, embedding = self.model(input_ids,segment_ids,input_mask,label_ids)
                
            # ---- parameters to keep track on ----#
            
            logits = logits.detach().cpu().numpy()
            #logits_array.append(logits)
            #embedding = embedding.detach().cpu().numpy()
            #embedding_array.append(embedding)
            label_ids = label_ids.to('cpu').numpy()

            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_instances += input_ids.size(0)
            nb_eval_steps +=1

            guid_char = [''.join(chr(i) for i in idi).strip('@') for idi in guid]
            guid_array.append(guid_char)
            tmp_wordpair_accuracy = (wordpair_accuracy(logits, guid_char, 'normal')/int(len(logits)/2))
            
            word_eval_accuracy += tmp_wordpair_accuracy

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_instances
        word_eval_accuracy = word_eval_accuracy/  nb_eval_steps

        result = {
                'test_loss': eval_loss,
                'test_accuracy': eval_accuracy,
                'wordpair_accuracy': word_eval_accuracy
                }
        logging.info("\r test loss = {:.5f} test accuracy={:.5f}  wordpair accuaracy={:.5f}"
                     .format(result['test_loss'],result['test_accuracy'],result['wordpair_accuracy'], end="\r"))

        output_eval_file = os.path.join(self.output_path,"test_result_{}_v2.json".format(version))
        output_history(output_eval_file, result)
        #self.save_result(logits_array=logits_array, guid_array=guid_array, embedding_array=embedding_array,
        #                 mode='model_variable', set_type='test')

    def word_choice_test(self):
        input_config_file = os.path.join(self.output_path, CONFIG_NAME)
        input_model_file = os.path.join(self.output_path, WEIGHTS_NAME)
        config = BertConfig(input_config_file)
        
        self.model = BertForSequenceClassification(config,
                                                   num_labels=self.num_labels,
                                                   similarity_mode=self.args.similarity)
        logging.info("model to test:{}".format(input_model_file))
        
        self.model.load_state_dict(torch.load(input_model_file, map_location=self.device[0]))
        
        self.model = torch.nn.DataParallel(self.model, device_ids=self.args.device_ids)
        pass
    
    def sentence_selection_simplified(self):
        
        logging.info("***** Running Simplified Entailment Sentence Selection *****")
        
        if self.args.new_mode:
            test_instances, instance_dict = self.get_instances(
                set_type='sentence_selection', setting=None)
            
            transpose = self.transpose_sentence_selection_instance(test_instances)
            del test_instances
            self.processor = None
            test_features = self.convert_instance_to_uniq_features(transpose, instance_dict,
                                                              'sentence_selection')
            
        else:
            logging.info("load features")
            instance_dict = self.load_instances_dict(set_type='sentence_selection', setting=None)
            transpose = pickle_load(self.output_path,'sentence_selection_transpose_{}.pickle'.format(self.version))
            #test_features = pickle_load(self.output_path,'sentence_selection_uniq_features.pickle')
            test_features = self.convert_instance_to_uniq_features(transpose, 
                                                               instance_dict,
                                                               'sentence_selection')
        self.load_model_default(setting=None)
        self.model.eval()
        
        result = {}
        
        for f in tqdm(test_features):
            input_ids = torch.tensor([f.input_ids])
            input_mask = torch.tensor([f.input_mask])
            segment_ids = torch.tensor([f.segment_ids])
            label_ids = torch.tensor([f.label_id])
            
            guid = ''.join(chr(char) for char in f.guid).strip('@') 
            
            with torch.no_grad():
                tmp_loss,logits,embedding = self.model(input_ids, segment_ids, input_mask, label_ids)
                
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()

            tmp_test_correct = entailment_accuracy(logits,label_ids)
            
            result[guid] = {
                'correct':tmp_test_correct,
                'logit':logits
            } 
            #print(result[guid])
        pickle_dump(
                self.output_path,
                'all_features_result_{}.pickle'.format(self.version),
                result)
        
        
        #result = pickle_load(self.output_path,'all_features_result_v2.pickle')
            
        batch_count = 0
        total = 0
        all_student_submit = []
        all_student_guid = []
        all_student_guid_array = []
        all_student_logits_array = []
        all_student_input_id_array = []

        current_student_incomplete = []
        all_student_guid_incomplete = []
        all_student_logits_incomplete = []
        all_student_input_id_incomplete = []
        
        
        question_num = 0
        
            
        sequence = [i.guid for i in transpose]
        test_batch_size = 800
        transpose_size = 24*self.args.sentence_selection_size/2
        
        batches = [sequence[x:x+test_batch_size] for x in range(0,len(transpose),test_batch_size)]
        batch_count = 0
        for batch in tqdm(batches, desc = "simplified sentence selection"):
            tmp_student_guid = []
            logits = []
            tmp_test_correct = []
            
            for single_guid in batch:
                tmp_student_guid = tmp_student_guid + [single_guid]
                logits = logits + list(result[single_guid]['logit'])
                tmp_test_correct = tmp_test_correct + list(result[single_guid]['correct'])
            
            if batch_count < 18:
                current_student_incomplete = current_student_incomplete+list(tmp_test_correct)
                all_student_guid_incomplete = all_student_guid_incomplete + tmp_student_guid
                all_student_logits_incomplete = all_student_logits_incomplete + list(logits)
            
                batch_count += 1
            
            if batch_count == 18:
                all_student_guid_array.append(all_student_guid_incomplete)
                all_student_logits_array.append(all_student_logits_incomplete)
            
                if len(all_student_submit) == 0:
                    all_student_submit = np.array(current_student_incomplete)

                    total += 1
                else:
                    all_student_submit = all_student_submit + np.array(current_student_incomplete)
                    total += 1
            
                current_student_incomplete = []
                all_student_guid_incomplete = []
                all_student_logits_incomplete = []
                question_num +=1
                batch_count = 0
                
                logging.info("total:{}".format(total))
                if total % transpose_size == 0 and total != 1:
                    logging.info("total:{}".format(total))
                    pickle_dump(
                        self.output_path,
                        'all_student_submit_{}_{}_simplified_{}.pickle'.format(
                            self.args.sentence_selection_size, total,self.version),
                        all_student_submit)
                    pickle_dump(
                        self.output_path,
                        'all_student_logits_array_{}_{}_simplified_{}.pickle'.format(
                            self.args.sentence_selection_size, total,self.version),
                        all_student_logits_array)
                    pickle_dump(self.output_path,
                        'all_student_guid_array_{}_{}_simplified_{}.pickle'.format(
                            self.args.sentence_selection_size, total,self.version),
                        all_student_guid_array)
                    all_student_submit = []
                    all_student_guid = []
                    all_student_guid_array = []
                    all_student_logits_array = []
                    all_student_input_id_array = []
                
            assert batch_count <= 18
            assert len(all_student_submit) <= 14400
       
        if len(all_student_guid_array) != 0:
            pickle_dump(
                self.output_path,
                'all_student_submit_{}_{}_simplified_final.pickle'.format(
                    self.args.sentence_selection_size, total),
                all_student_submit)
            pickle_dump(
                self.output_path,
                'all_student_logits_array_{}_{}__simplified_final.pickle'.format(
                    self.args.sentence_selection_size, total),
                all_student_logits_array)
            pickle_dump(self.output_path,
                'all_student_guid_array_{}_{}__simplified_final.pickle'.format(
                    self.args.sentence_selection_size, total),
                all_student_guid_array)
            

    
    def sentence_selection(self):
        logging.info("***** Running Entailment Sentence Selection *****")
        self.load_model_default(setting=None)
        
        if self.args.new_mode:
            test_instances, instance_dict = self.get_instances(
                set_type='sentence_selection', setting=None)
            
            transpose = self.transpose_sentence_selection_instance(test_instances)
            del test_instances
            self.processor = None
            test_features = self.convert_instance_to_features(transpose, instance_dict,
                                                              'sentence_selection')
            
        else:
            logging.info("load features")
            instance_dict = self.load_instances_dict(set_type='sentence_selection', setting=None)
            test_features = self.load_features(set_type='sentence_selection', setting=None)
        
        
        logging.info("convert dataframe")
        #test_batch_size = 1200
        test_batch_size = 800
        
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        all_guid = torch.tensor([f.guid for f in test_features], dtype=torch.long)
        
        test_features = None

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_guid)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)
        batch_count = 0
        total = 0
        all_student_submit = []
        all_student_guid = []
        all_student_guid_array = []
        all_student_logits_array = []
        all_student_input_id_array = []

        current_student_incomplete = []
        all_student_guid_incomplete = []
        all_student_logits_incomplete = []
        all_student_input_id_incomplete = []
        self.model.eval()
        
        question_num = 0
        
        for batch in tqdm(test_dataloader, 'sentence selection') :
            batch = tuple(t.to(self.device[0]) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, guid = batch
            guid = list(guid.detach().cpu().numpy())
            
            tmp_student_guid = [''.join(chr(i) for i in idi).strip('@') for idi in guid]
            
            logging.info(tmp_student_guid)
            with torch.no_grad():
                tmp_test_loss,logits,embedding = self.model(input_ids,segment_ids,input_mask,label_ids)
                
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            logging.info(logits)
            
            tmp_test_correct = entailment_accuracy(logits,label_ids)
            logging.info(tmp_test_correct)
            sys.exit(1)
            if batch_count ==0:
                save_guid = tmp_student_guid[0]
            if batch_count < 18:
                current_student_incomplete = current_student_incomplete+list(tmp_test_correct)
                all_student_guid_incomplete = all_student_guid_incomplete + tmp_student_guid
                all_student_logits_incomplete = all_student_logits_incomplete + list(logits)
            #   all_student_input_id_incomplete = all_student_input_id_incomplete + list(input_ids)
                batch_count += 1
                #print(len(current_student_incomplete))
                #print(batch_count)
            if batch_count == 18:
                all_student_guid_array.append(all_student_guid_incomplete)
                all_student_logits_array.append(all_student_logits_incomplete)
            #    all_student_input_id_array.append(all_student_input_id_incomplete)

                if len(all_student_submit) == 0:
                    all_student_submit = np.array(current_student_incomplete)

                    total += 1
                else:
                    all_student_submit = all_student_submit + np.array(current_student_incomplete)
                    total += 1
                    # break
                current_student_incomplete = []
                all_student_guid_incomplete = []
                all_student_logits_incomplete = []
                question_num +=1
                batch_count = 0
                
                logging.info("total:{}".format(total))
                if total % 100 == 0 and total != 1:
                    logging.info("total:{}".format(total))
                    pickle_dump(
                        self.output_path,
                        'all_student_submit_{}_{}.pickle'.format(
                            self.args.sentence_selection_size, total),
                        all_student_submit)
                    pickle_dump(
                        self.output_path,
                        'all_student_logits_array_{}_{}.pickle'.format(
                            self.args.sentence_selection_size, total),
                        all_student_logits_array)
                    pickle_dump(self.output_path,
                        'all_student_guid_array_{}_{}.pickle'.format(
                            self.args.sentence_selection_size, total),
                        all_student_guid_array)
                    all_student_submit = []
                    all_student_guid = []
                    all_student_guid_array = []
                    all_student_logits_array = []
                    all_student_input_id_array = []
                
            assert batch_count <= 18
            assert len(all_student_submit) <= 14400
            #if total == 4:
            #    break
           
        if len(all_student_guid) != 0:
            pickle_dump(
                self.output_path,
                'all_student_submit_{}_{}_final.pickle'.format(
                    self.args.sentence_selection_size, total),
                all_student_submit)
            pickle_dump(
                self.output_path,
                'all_student_logits_array_{}_{}_final.pickle'.format(
                    self.args.sentence_selection_size, total),
                all_student_logits_array)
            pickle_dump(self.output_path,
                'all_student_guid_array_{}_{}_final.pickle'.format(
                    self.args.sentence_selection_size, total),
                all_student_guid_array)
       
        '''

        max_index = np.where(all_student_submit == max(all_student_submit))[0]
        
        
        best_example_guid_array = []
        best_example_guid = []
        for col_index in max_index:
            best_example_guid = [guid_list[col_index][0:3] for guid_list in all_student_guid_array]
            best_example_guid_array.append(best_example_guid)

        with codecs.open(
            os.path.join(
                MODEL_ROOT,
                self.args.output_dir,
                "entailment_sentence_selection_{}_v2.txt".format(args.sentence_selection_size)),
            'w',encoding='utf-8'
        ) as outfile:
            for best_pair in best_example_guid_array:
                uniq_best = set(best_pair)
                outfile.write("{}\n".format(uniq_best))
       '''

    def behavior_check(self, behavior_setting):
        logging.info("***** Running Behavior Check *****")
        self.load_model_default()
        
        instances_dictionary = {}
        instance_dict_dictionary = {}
        features_dictionary = {}
        
        for i in behavior_setting:
            logging.info("Running Behavior Check {} ".format(i))
            instances_dictionary[i], instance_dict_dictionary[i] = self.get_instances(
                set_type='behavior_check', setting=i)
            features_dictionary[i] = self.convert_instance_to_features(
                instances_dictionary[i], instance_dict_dictionary[i], i)
            all_input_ids = torch.tensor([f.input_ids for f in features_dictionary[i]], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features_dictionary[i]], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features_dictionary[i]], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in features_dictionary[i]], dtype=torch.long)

            all_guid = torch.tensor([f.guid for f in features_dictionary[i]], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids,all_input_mask,all_segment_ids,all_label_ids,all_guid)

            eval_sampler = SequentialSampler(eval_data)
            
            questions_per_id = 5
            # 6 (number of sentence, 3 for each) * 4(number of entailment type)
            batch_size = 24 * questions_per_id
            logging.info("behavior check batch size:{}".format(batch_size))
            eval_dataloader = DataLoader(eval_data,sampler=eval_sampler, batch_size = batch_size)

            self.model.eval()

            guid_array = []
            logits_array = []
            flag = 0
            
            for input_ids, input_mask, segment_ids, label_ids, guid in tqdm(eval_dataloader,desc="Behavior Check"):
                input_ids = input_ids.to(self.device[0])
                input_mask = input_mask.to(self.device[0])
                segment_ids = segment_ids.to(self.device[0])
                label_ids = label_ids.to(self.device[0])

                with torch.no_grad():
                    tmp_eval_loss, logits, embedding=self.model(input_ids, segment_ids, input_mask, label_ids)
                    
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                tmp_eval_accuracy = accuracy(logits, label_ids)

                eval_loss = tmp_eval_loss.mean().item()
                eval_accuracy = tmp_eval_accuracy / input_ids.size(0)
                
                guid_char = [''.join(chr(i) for i in idi).strip('@') for idi in guid]

                tmp_wordpair_accuracy = wordpair_accuracy(logits,guid_char,i)

                # if flag < 10:
                #    logging.info(logits)
                #    logging.info(guid_char)
                #    flag += 1

                guid_array.append(guid_char)
                logits_array.append(list(logits))
                word_eval_accuracy = tmp_wordpair_accuracy/ int(len(logits)/2)
                result = {
                    'eval_loss': eval_loss,
                    'eval_accuracy': eval_accuracy,
                    'wordpair_accuracy': word_eval_accuracy
                }

                if i == 'normal':
                    output_eval_file = os.path.join(self.output_path, "behavior_normal_eval_result_v3.json")
                elif i == 'reverse':
                    output_eval_file = os.path.join(self.output_path, "behavior_reverse_eval_result_v3.json")
                elif i == 'double_reverse':
                    output_eval_file = os.path.join(self.output_path, "behavior_double_reverse_eval_result_v3.json")
                elif i == 'dropout':
                    output_eval_file = os.path.join(self.output_path, "behavior_dropout_eval_result_v3.json")
                pickle_dump(self.output_path, 'behavior_guid_test.pickle', guid_array)
                pickle_dump(self.output_path, 'behavior_logits.pickle', logits_array)
            #with open(output_eval_file, 'w', encoding='utf-8') as outfile:
            #    json.dump({}, outfile, indent=4)
                #
                #with open(output_eval_file,"a+") as infile:
                #logging.info("**** Eval result ****")
                #for key in sorted(result.keys()):
                    #logging.info("  %s = %s", key, str(result[key]))
                output_history(output_eval_file, result)
        
    def save_whole_model(self, epoch, loss, global_step):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self

        model_path = os.path.join(self.output_path, "model_{:0>5}.net".format(global_step))
        torch.save({
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss": loss
        },model_path)
        
        output_config_file = os.path.join(self.output_path, "model_{:0>5}_config.net".format(global_step))
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

    def load_whole_model(self, version):
        input_config_file = os.path.join(self.output_path, 'config.json')
        config = BertConfig(input_config_file)
        self.model = BertForSequenceClassification(
            config, num_labels=self.num_labels, similarity_mode=self.args.similarity)
        input_model_file = os.path.join(self.output_path, 'model_{}.net'.format(version))
        checkpoint = torch.load(input_model_file, map_location=self.device[0])

        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.to(self.device[0])
        self.model = torch.nn.DataParallel(self.model, device_ids=self.args.device_ids)
        
    def save_model_state_dict(self, setting):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
        
        if setting != 'word_choice':
            logging.info('save default')
            output_model_file = os.path.join(self.output_path, WEIGHTS_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
            output_config_file = os.path.join(self.output_path, CONFIG_NAME)

        else: 
            output_model_file = os.path.join(self.output_path, 
                                             'word_choice_model_v2_{}.bin'.format(self.args.learning_rate))
            output_config_file = os.path.join(self.output_path, 
                                              'word_choice_bert_config_v2_{}.json'.format(self.args.learning_rate))
            
        logging.info(output_model_file)
        torch.save(model_to_save.state_dict(), output_model_file)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
        
    def load_model_default(self,setting=None):
        
        if setting != 'word_choice':
            input_model_file = os.path.join(self.output_path, WEIGHTS_NAME)
            input_config_file = os.path.join(self.output_path, CONFIG_NAME)
        else:
            input_model_file = os.path.join(
                self.output_path, 'word_choice_model_v2_{}.bin'.format(self.args.learning_rate))
            input_config_file = os.path.join(
                self.output_path, 'word_choice_bert_config_v2_{}.json'.format(self.args.learning_rate))
        logging.info(input_model_file)
        config = BertConfig(input_config_file)
        self.model = BertForSequenceClassification(
            config, num_labels=self.num_labels, similarity_mode=self.args.similarity)
        self.model.load_state_dict(torch.load(input_model_file, map_location=self.device[0]))
        self.model.to(self.device[0])
        self.model = torch.nn.DataParallel(self.model, device_ids=self.args.device_ids)

    def save_result(self, mode, epoch=None, step=None, loss=None, start_time=None, end_time=None,
                    logits_array=None, guid_array=None, embedding_array=None, set_type=None):
        # save_model_result
        if mode == 'model_result':
            logging.info('epoch:{} batch:{} loss:{}'.format(epoch,step,loss))
            result = {
                "epoch":epoch,
                "batch":step,
                "loss":loss
            }
            output_train_file = os.path.join(MODEL_ROOT,self.args.output_dir, "train_result.json")
            output_history(output_train_file,result)
        
        # save model_output
        if mode == 'model_time':
            with codecs.open(os.path.join(self.output_path, 'input_config.json'), 'a+') as outfile:
                outfile.write('Elapsed time:{}'.format(end_time-start_time))
        if mode == 'model_variable':
            pickle_dump(self.output_path,
                        '{}_logits_array_v2_{}.pickle'.format(set_type,self.args.learning_rate),logits_array)
            pickle_dump(self.output_path,
                        '{}_guid_array_v2_{}.pickle'.format(set_type,self.args.learning_rate),guid_array)
            pickle_dump(self.output_path,
                        '{}_embedding_array_v2_{}.pickle'.format(set_type,self.args.learning_rate),
                        embedding_array[:1250])


def dict_tokenizer(tokenizer,wordpair_dict,instance_dict,indexer):
    """ return a new_instance_dict with tokens and target word index
    input instance dict and pre-tokenized the row sentences and find the corresponding target word index
    """
    #tmp_tuple = []
    new_instance_dict={}
    for pair in wordpair_dict:
        new_instance_dict[pair]={}
    
        for count,key in enumerate(instance_dict[pair]):
            prefix = key[0:2]
            if prefix == '11' or prefix == '21':
                keyword = wordpair_dict[pair][0]
            else:
                keyword = wordpair_dict[pair][1]

            raw_sentence = instance_dict[pair][key]
            #print(key)
            #print(raw_sentence)
            try:tokens = tokenizer.tokenize(raw_sentence)
            except: 
                
                logging.warning(raw_sentence)
                sys.exit(0)
            #tokens = list(filter(None,tokens[:-2])) old data
            tokens = list(filter(None,tokens))
            
            try:
                index = indexer.find_index_single_case(keyword,tokens)
            except:
                logging.warning(keyword,tokens)
                sys.exit(0)
            tmp_type = [raw_sentence,tokens,index]
            new_instance_dict[pair][key] = tmp_type
            #if count<10:
            #    print(tokens,index)
    return new_instance_dict


def accuracy(output,labels):
    outputs = np.argmax(output,axis=1)
    return np.sum(outputs==labels)


def entailment_accuracy(logits,label):
    outputs = np.argmax(logits,axis=1)
    return (outputs == label).astype(np.float32)


def wordpair_accuracy(logits,guid,set_type):
    #set_type = self.set_type
    length = len(logits)
    indices = list(range(length))
    split_index = [ i for i in indices if i%4 ==0]
    wordpair_result = []
    #logging.info(set_type)
    #suffix_label = []
    for index in split_index:
        #TODO: guid prefix, if prefix == 1, check 1,3 otherwise check 2,4
        prefix = guid[index].split('-')[0][0:2]
        suffix = guid[index].split('-')[1][0:2]
        #logging.info(prefix)
        #logging.info(suffix)


        if prefix == '11' or prefix == '12':
            if logits[index][0]> logits[index+2][0]:
                wordpair_result.append(0)
            else:
                wordpair_result.append(1)
            if logits[index+1][1] > logits[index+3][1]:
                wordpair_result.append(1)
            else:
                wordpair_result.append(0)
        elif prefix == '22' or prefix == '21':
            if logits[index][1]> logits[index+2][1]:
                wordpair_result.append(0)
            else:
                wordpair_result.append(1)
            if logits[index+1][0] > logits[index+3][0]:
                wordpair_result.append(1)
            else:
                wordpair_result.append(0)

    if set_type== 'normal' or set_type=='reverse':
        label = []
        for i in range(int(len(indices)/4)):
            label.append(0)
            label.append(1)
    else:
        label = []
        for i in range(int(len(indices)/4)):
            label.append(1)
            label.append(0)

    #logging.info(label)
    #logging.info(wordpair_result)
    #logging.info(np.sum(np.array(wordpair_result)==np.array(label)))
    #sys.exit(1)

    return np.sum(np.array(wordpair_result)==np.array(label))
    #return wordpair_result, label
