# enoding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""
from __future__ import absolute_import, division, print_function
import os
import torch
from definition import MODEL_ROOT
from utils import create_json_output_file as create_empty_json_file
from utils import json_dump
import logging
logging.getLogger().setLevel(logging.INFO)

def check_if_directory_exist(args,output_path):
    if (os.path.exists(output_path) and os.listdir(output_path) 
            and args.do_train and args.new_mode):
        raise ValueError("Output directory ({}) already exists and is not empty."
                         .format(args.output_dir))
    if not os.path.exists(output_path):
        logging.info("CREATE MODEL DIRECTORY")
        os.makedirs(output_path)
        cache_path = os.path.join(output_path,'cache')
        os.makedirs(cache_path)


def clean_old_files(args,output_path):
    if args.do_train:
        create_empty_json_file(output_path,"train_result.json")
    if args.do_word_choice_train:
        create_empty_json_file(output_path,"word_choice_result.json")
    if args.do_eval and args.dropout:
        create_empty_json_file(output_path,"dev_dropout_eval_result.json")
    elif args.do_eval:
        create_empty_json_file(output_path,"dev_eval_result.json")
    if args.do_test or args.do_test_more:
        create_empty_json_file(output_path,"test_eval_result.json")
    if args.do_word_choice_test:
        create_empty_json_file(output_path,"word_choice_test_result.json")
    if args.normal: 
        create_empty_json_file(output_path,"behavior_normal_eval_result_v3.json")
    if args.reverse:
        create_empty_json_file(output_path,"behavior_reverse_eval_result_v3.json")        
    if args.double_reverse:
        create_empty_json_file(output_path,"behavior_double_reverse_eval_result_v3.json")


def run(args):
    # ----- CUDA SETTING -----#
    
    device = [torch.device('cuda:{}'.format(x)) for x in args.device_ids]
    
    n_gpu = torch.cuda.device_count()
    
    logging.info("device: {} n_gpu: {}".format(device, n_gpu))
    
    # ----- ENVIRONMENT SETTING -----#
    output_path =os.path.join(MODEL_ROOT,args.output_dir)
    check_if_directory_exist(args,output_path)
    clean_old_files(args,output_path)
    config = vars(args)
    json_dump(output_path,'config.json',config)

     #----- CREATE ALL CLASS OBJECT -----#
        
    if args.entailment:
        from EntailmentModel import EntailmentModel
        #version = "v2_g1"
        entailment_model = EntailmentModel(args,device,args.version)
        
        if args.do_train:
            
            entailment_model.train()
            
        if args.do_word_choice_train:
            
            #entailment_model.word_choice_train()
            pass
        if args.do_eval:
            pass
        if args.do_test:
            #version = '02501'
            entailment_model.test() 
        if args.do_behavior_check:
            setting = []
            if args.normal:
                setting.append('normal')
            if args.reverse:
                setting.append('reverse')
            if args.double_reverse:
                setting.append('double_reverse')
            entailment_model.behavior_check(setting)
            
        if args.do_sentence_selection:
            entailment_model.sentence_selection_simplified()
       
    
    else:
        from FITBModel import FITBModel
        fitb_model = FITBModel(args,device)
        
        if args.do_train:
            if args.cross_validation:
                fitb_model.cv_train(args.k_folds)
            else:
                fitb_model.train()

        if args.do_test:
            if args.version is not None:
                fitb_model.test(args.version)
            else:
                fitb_model.test()
        
        if args.do_behavior_check:
            setting = []
            if args.normal:
                setting.append('normal')
            if args.reverse:
                setting.append('reverse')
            if args.double_reverse:
                setting.append('double_reverse')
            fitb_model.behavior_check(setting)
             
        if args.do_sentence_selection:
            fitb_model.sentence_selection()
