import os
import json
import codecs
import logging
import sys
logging.getLogger().setLevel(logging.INFO)
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data")
MODEL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),"model")

GLOBAL_INDEX_PATH = os.path.join(DATA_ROOT, 'GLOBAL_INDEX.json')
        
with codecs.open(GLOBAL_INDEX_PATH, 'r', encoding='utf-8') as infile:
    GLOBAL_INDEX = json.load(infile)
        
WORD_TO_ID_PATH = os.path.join(DATA_ROOT, 'word_to_id.json')
with codecs.open(WORD_TO_ID_PATH, 'r', encoding='utf-8') as infile:
    WORD_TO_ID = json.load(infile)
        
WORD_MAP_PATH = os.path.join(DATA_ROOT, 'inflection_set.json')
with codecs.open(WORD_MAP_PATH, 'r', encoding='utf-8') as infile:
    WORD_MAP = json.load(infile)
