# Assessing the Helpfulness of Learning Materials with Inference-Based Learner-Like Agent

Code for our EMNLP 2020 paper, ["Assessing the Helpfulness of Learning Materials with Inference-Based Learner-Like Agent"](https://arxiv.org/abs/2010.02179).


### Introduction:
In the language learning process, many English-as-a-second language learners often have trouble using near-synonym words correctly, and often look for example sentences to learn how two nearly synonymous terms differ. In automatic example extraction research, prior work uses hand-crafted scores to recommend sentences but suffers from the complexity of human-designed scoring functions. We proposed to use an inference-based learner-like agent to mimic human behavior on learning from example sentences. This agent leverages entailment modeling to connect the question context and the correct answer.
Experimental results show that the proposed agent is equipped with good learner-like behavior to achieve the best performance in both fill-in-the-blank (FITB) and good example sentence selection tasks. We further deploy the proposed model to college ESL learners. The results of this study show that inference enables the ability to choose easy and helpful examples. Compared to other models, the proposed agent improves the score of more than 17\% of students in the post-test.


### Installation
1. Clone this repository
```
git clone https://github.com/joyyyjen/Inference-Based-Learner-Like-Agent.git
```
2. Create python environment 
```
virtualenv --python=/usr/bin/python3.6 emnlp_lla_env
```
3. Install required packages
```
pip3 install -r requirements.txt 
```

4. Download dataset

unzip the data.zip in to a directory called"data"

### Training and Testing

#### Training

```
python3.6 __main__.py --wordpair common,ordinary --output_dir model_05_FITB --warmup_proportion 0.3 --num_train_epochs 2 --bert_model bert-base-uncased --do_train --local_rank 2 --device_ids 0 1 --multi --seed 4 --new_mode --entailment 
```
To train a context-based learner-like agent, remove --entailment arguments from the commend above. 

#### Testing

``` 
python3.6 __main__.py --wordpair common,ordinary --output_dir model_05_FITB --warmup_proportion 0.3 --num_train_epochs 2 --cache_dir cache --bert_model bert-base-uncased --do_test --local_rank 2 --device_ids 0 1 --multi --seed 4
```

### Experiments
#### Behavior Check

```
python3.6 __main__.py --entailment --wordpair {word-pair} --output_dir {model_path} --do_behavior_check --normal --reverse --double_reverse
```

#### Sentence Selection

```
python3.6 __main__.py --entailment --wordpair {word-pair} --output_dir {model_path} --do_sentence_selection --sentence_selection_size 50
```            
          
### Dataset

**Near-Synonym Dataset** is a dataset contains 30 English word-pairs along with 8000 sentences for training, 2000 sentences for evaluation. 

The data is extracted from the Wikipedia sentences on January 20, 2020, and has been carefully formated to fit our model need. 

The dataset is separated into 30 files. Each file is named "WIKI_DATABASE_{word-pair}.json".

In each data file, each sentence has its word-pair id, sentence_id, and word_id, in which sentence id and word id are pre-defined in GLOBAL_INDEX.json and word_to_id.json.

Those are two instances in WIKI_DATABASE_small|little.json
```
"7_57_1": {
                "word_id": 57,
                "sentence_id": 1,
                "rawSentence": "The sparse human population is largely nomadic, with some livestock, mostly small ruminants and camels.\n",
                "targetWord": "small"
                }
"7_52_1122": {
                "word_id": 52,
                "sentence_id": 1122,
                "rawSentence": "Finally, in April 1944, leukemia was diagnosed, but by this time, little could be done .\n",
                "targetWord": "little"
            }
```
Target words from raw sentences are replaced by a "[MASK]" using module "multiInstanceProcessor" when we load a choosen word-pair data. 


**The Near-Synonym Example Extraction Evaluation Dataset** is annotated by an EFL expert to find the best examples for learners to learn the near-synonym difference. It consists of 600 sentences, 20 sentences for each wordpair, and 10 sentences for each word.

The filename is "annotated_evaluation_dataset.tsv".
Column one is a target word from a word-pair, column two is a raw sentence, and column three is the expert choice.(Y is accpet and N is reject) 

For instance, a row can look like the following:
```
task	After being liberated, he left Cuba and headed to Costa Rica, where he dedicated his time to pedagogical tasks, and to his activities as leader of the PRD.	Y
```


### Citation
```
If you find our work useful in your research, please consider citing: 
@misc{jen2020assessing,
      title={Assessing the Helpfulness of Learning Materials with Inference-Based Learner-Like Agent}, 
      author={Yun-Hsuan Jen and Chieh-Yang Huang and Mei-Hua Chen and Ting-Hao 'Kenneth' Huang and Lun-Wei Ku},
      year={2020},
      eprint={2010.02179},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
