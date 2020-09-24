# learner-like-agent
## Update on Feb 4th 2020


Fill-in-the-blank and entailment learner-like agent model for suggesting useful example sentences

#### version
python3.6

#### requirements
see requirements package for more information

##### execution
```
python3.6 __main__.py --wordpair common,ordinary --output_dir model_05_FITB --warmup_proportion 0.3 --num_train_epochs 2 --cache_dir cache --bert_model bert-base-uncased --do_train --local_rank 2 --device_ids 0 1 --multi --seed 4 --new_mode
```

#### Modules
**0.definitin.py**
Custom data root, model root, output root, and other project path

**1. run.py:**
Call corresponding Model and its methods

**2. FITBModel.py**
Create FITB Model for training, validation, and testing

**3. EntailmentModel.py:**
Create Entailment Model for training, validation, and testing. It includes word_choice training. 

**4. DataProcess_v2.py:**
Methods to read, parse data

**5. multiInstanceProcessor_v2.py:**
Methods to create model instances

**6. Features.py:**
Methods to convert instance to bert features

**7. utils.py:**
utils for finding index and other data loading or exporting method

**8. Analysis.py:**
behavior check analysis: compute t-score and draw behavior graph

**9. EntailmentSentenceSelection.py:**