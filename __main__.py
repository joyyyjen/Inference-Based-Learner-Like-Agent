import argparse
from run import *


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wordpair", dest="wordpair", 
            help="wordset, use ',' to separate.", default="accept,agree")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
            help="Bert pre-trained model selected in the list: bert-base-uncased, "
            "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
            "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",default=None,type=str,required=True,
            help="The output directory where the model predictions and checkpoints will be written.")
    
    ## Other parameters
    parser.add_argument("--save_model_freq",default=10,type=int)
    parser.add_argument("--version",default=None,type=str)

    parser.add_argument("--cache_dir",default="cache",type=str,
            help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",default=256,type=int,
            help="The maximum total input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.")
    parser.add_argument("--cross_validation",action='store_true')
    parser.add_argument("--k_folds",default=5,type=int)

    parser.add_argument("--do_train",action='store_true',
            help="Whether to run training.")
    parser.add_argument("--do_eval",action='store_true',
            help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_simi",action='store_true',
            help="Whether to run eval on the dev set.")
    
    parser.add_argument("--do_word_choice_train",action='store_true',
            help="Whether to run word choice finetune.")
    parser.add_argument("--do_word_choice_eval",action='store_true',
            help="Whether to run word choice evaluation.")
    parser.add_argument("--do_word_choice_test",action='store_true',
            help="Whether to run word choice testing.")
    
    
    parser.add_argument("--do_behavior_check",action='store_true',
            help="Whether to run behavior_check on the dev set.")
    parser.add_argument("--normal",action='store_true')
    parser.add_argument("--reverse",action='store_true')
    parser.add_argument("--double_reverse",action='store_true')
    
    parser.add_argument("--do_sentence_selection",action='store_true',
            help="Whether to run sentence_selection.")
    parser.add_argument("--sentence_selection_size",default=100,type=int,
            help="Total batch size for eval.")
    parser.add_argument("--example_set_size",default=3,type=int,required=True)
    
    parser.add_argument("--do_test",action='store_true',
            help="Whether to run test on the test test")
    parser.add_argument("--do_test_more",action='store_true',
            help="Whether to run test on the test test")
    parser.add_argument("--new_mode",action='store_true')
    parser.add_argument("--uniq",action='store_true')
    parser.add_argument("--sBert",action='store_true')
    parser.add_argument("--similarity",action='store_true')
    parser.add_argument("--multi",action='store_true')
    parser.add_argument("--dropout",action='store_true')
    
    
    parser.add_argument("--entailment",action='store_true')
    
    parser.add_argument("--do_lower_case",action='store_true',
            help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",default=32,type=int,
            help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",default=8,type=int,
            help="Total batch size for eval.")
    parser.add_argument("--learning_rate",default=5e-5,type=float,
            help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",default=3.0,type=float,
            help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",default=0.1,type=float,
            help="Proportion of training to perform linear learning rate warmup for. "
            "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",action='store_false',
            help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",type=int,default=-1,
            help="local_rank for distributed training on gpus")
    parser.add_argument("--device_ids",nargs ='+',type=int, required=True,
            help= "cuda device ids")
    parser.add_argument('--seed',type=int,default=42,
            help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',type=int,default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    
    
    return parser.parse_args()


def main():
    
    args = parse_arg()
    run(args)

if __name__ == "__main__":
    main()
