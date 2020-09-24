from definition import WORD_MAP
from utils import json_load
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

import os
import sys
import math
import logging
logging.getLogger().setLevel(logging.INFO)


class FindIndex(object):
    def __init__(self, model_path):
        self.model_path = model_path

    def get_transform_mapping_list(self, keyword):
        """
        A helper function
        to retrieve list form of transform mapping given keyword during findSingleCaseS
        """
        transform_mapping = json_load(os.path.join(self.model_path, 'cache'), 'mapping.json')

        bow = {}
        for word in transform_mapping:
            try:
                bow[word] = list(transform_mapping[word].keys())
            except:
                print(transform_mapping[word])
                bow[word] = transform_mapping[word]

        # print(bow)
        return bow[keyword]

    @staticmethod
    def get_pos_list(keyword):
        if keyword in WORD_MAP:
            return [keyword, WORD_MAP[keyword]]
        else:
            return [keyword]

    def get_forms(self, keyword):
        stemmer = SnowballStemmer("english")

        transform_mapping = json_load(os.path.join(self.model_path, 'cache'), 'mapping.json')

        bow = {}
        for word in transform_mapping:
            bow[word] = list(transform_mapping[word])
            bow[word].append(stemmer.stem(word))

        return bow[keyword]

    @staticmethod
    def word_retokenize(tokens, word):
        """
        A helper function to deal with bert tokenizer
        """
        stemmer = SnowballStemmer("english")
        lemmatizer = WordNetLemmatizer()

        target = ""
        index = []
        for i in range(len(tokens)):
            if len(index) != 0 and '##' not in tokens[i]:
                target_lemma = lemmatizer.lemmatize(target)
                target_stem = stemmer.stem(target)
                word_stem = stemmer.stem(word)

                if target_lemma == word or target_stem == word_stem:
                    return index
                else:
                    index = []
                    target = ""
                #    logger.warning("Target Unfound Try 2")
                #    sys.exit(1)
            if '##' in tokens[i]:
                if len(index) > 1:
                    target = target + tokens[i][2:]
                    index.append(i)
                elif len(index) == 0:
                    target = tokens[i - 1] + tokens[i][2:]
                    index.append(i - 1)
                    index.append(i)
            if '##' in tokens[i] and i + 1 == len(tokens):
                target_lemma = lemmatizer.lemmatize(target)
                target_stem = stemmer.stem(target)
                word_stem = stemmer.stem(word)
                if target_lemma == word or target_stem == word_stem:
                    return index
        '''
        logging.warning("Target Unfound")
        print("target word",word)
        print("tokens:",tokens)
        '''
        return None
        # sys.exit(1)

    def find_index_single_case(self, keyword, tokens):
        """
        A helper function to find index for a single sentence
        """
        found = False

        pos_list = self.get_pos_list(keyword)

        for word in pos_list:
            if word in tokens:
                target_index = tokens.index(word)
                found = True
        if not found:
            target_index = self.word_retokenize(tokens, keyword)
            if target_index is not None:
                found = True

        if not found:
            pos_list = self.get_forms(keyword)
            for word in pos_list:
                if word in tokens:
                    target_index = tokens.index(word)
                    found = True

        if not found:
            stemmer = SnowballStemmer("english")
            stem_tokens = [stemmer.stem(w) for w in tokens]
            for word in pos_list:
                if word in stem_tokens:
                    target_index = stem_tokens.index(word)
                    found = True
        if not found:
            logging.warning('find index single case error')
            print("keyword", keyword)
            print(pos_list)
            print(tokens)
            sys.exit(1)
        return target_index

    @staticmethod
    def replace_target_word_with_mask(tokens_b, target_index_b, mode=None):
        """ mode = True
        when if no [MASK] substitution
        """

        original_b = ""
        if '[MASK]' in tokens_b:
            original_b = tokens_b[target_index_b]

            return tokens_b, target_index_b, original_b
        if type(target_index_b) == int:

            original_b = tokens_b[target_index_b]

            if mode is None:
                tokens_b[target_index_b] = '[MASK]'

        else:

            for i in target_index_b:
                original_b = original_b + tokens_b[i].strip('##')
            if mode is None:
                tokens_b[target_index_b[0]] = '[MASK]'

                length = len(target_index_b) - 1
                i = 0
                while True:
                    if i == length:
                        break
                    tokens_b.pop(target_index_b[0] + 1)
                    i += 1
                    target_index_b.pop()
            target_index_b = target_index_b[0]

        try:
            assert target_index_b <= len(tokens_b) and target_index_b >= 0
        except:
            logging.warning("target b out of range")
            print(target_index_b, tokens_b)
            sys.exit(1)
        return tokens_b, target_index_b, original_b

    @staticmethod
    def get_avg_list(reminder, avg_list):
        for i in range(len(avg_list)):
            if (reminder < 1 and i != 0) or (reminder == 0):
                break
            avg_list[i] = avg_list[i] + 1
            reminder = reminder - 1

        return avg_list

    def truncate(self, list_to_index, max_length):

        while True:
            smaller_than_avg = []
            larger_than_avg = {}

            length_list = [list_to_index[i][2] for i in list_to_index]
            index_list = [list_to_index[i][1] for i in list_to_index]

            try:
                length_list_double = [len(list_to_index[i][0]) for i in list_to_index]
                assert length_list_double == length_list
            except:
                logging.warning("Initial Truncate Length Mismatch")
                print("length_list:", length_list)
                print("Calculated:", length_list_double)
                sys.exit(1)

            # print(sum(length_list))
            if sum(length_list) <= max_length:
                break
            else:
                avg = max_length / len(index_list)
                reminder = max_length % len(length_list)
                # print(avg)
                # print(reminder)
                for i in list_to_index:
                    if list_to_index[i][2] < avg:
                        smaller_than_avg.append(list_to_index[i][2])
                    else:
                        larger_than_avg[i] = (list_to_index[i])
                    # larger_than_avg_index.append(j)

                # print(larger_than_avg)

                truncate_length = max_length - sum(smaller_than_avg)
                # print(truncate_length)
                truncate_avg = truncate_length / len(larger_than_avg)
                # print(truncate_avg)
                truncate_reminder = truncate_length % len(larger_than_avg)
                # print(truncate_reminder)
                truncate_avg_list = [math.floor(truncate_avg)] * len(larger_than_avg)
                truncate_avg_list = self.get_avg_list(truncate_reminder, truncate_avg_list)
                # print(truncate_avg_list)

                for new_avg, name in zip(truncate_avg_list, larger_than_avg):
                    length = larger_than_avg[name][2]
                    index = larger_than_avg[name][1]
                    if new_avg > length:
                        continue
                    elif index + math.ceil(new_avg / 2) < length and index - math.ceil(new_avg / 2) > 0:
                        # print("in range")
                        # print(index - math.floor(new_avg/2),index,index + math.floor(new_avg/2))
                        larger_than_avg[name][2] = index + math.floor(new_avg / 2) - (index - math.ceil(new_avg / 2))
                        list_to_index[name][0] = list_to_index[name][0][(index - math.ceil(new_avg / 2)):index] + \
                                                 list_to_index[name][0][index:(index + math.floor(new_avg / 2))]
                        assert larger_than_avg[name][2] == len(list_to_index[name][0])

                    elif index + math.ceil(new_avg / 2) >= length:
                        # print("left range")
                        # print(new_avg,index,length)
                        larger_than_avg[name][2] = new_avg
                        list_to_index[name][0] = list_to_index[name][0][(length - new_avg):index] + list_to_index[name][
                                                                                                        0][index:]
                        # print(len(list_to_index[name][0]))
                        assert larger_than_avg[name][2] == len(list_to_index[name][0])
                    elif index - math.ceil(new_avg / 2) <= 0:
                        # print("right range")
                        # print(0,index,new_avg)
                        larger_than_avg[name][2] = new_avg
                        list_to_index[name][0] = list_to_index[name][0][:index] + list_to_index[name][0][index:new_avg]
                        assert larger_than_avg[name][2] == len(list_to_index[name][0])
                    else:
                        print("truncate others")
                        print(name)
                        print(new_avg)
                        print(list_to_index[name])
                    # print(sum([list_to_index[i][2] for i in list_to_index]))
        try:
            length_list = [len(list_to_index[i][0]) for i in list_to_index]
            assert sum(length_list) == sum([list_to_index[i][2] for i in list_to_index])
        except:
            logging.warning("Truncate Length Mismatch")
            print("length_list len:", sum(length_list))
            print("Marked:", sum([list_to_index[i][2] for i in list_to_index]))
            sys.exit(1)

        try:
            assert sum([list_to_index[i][2] for i in list_to_index]) <= max_length

        except:
            logging.warning("larger than max_length")
            print(list_to_index)
            sys.exit(1)
        # Forgot what this part is for and what 20 is   
        # try:
        #     assert sum([list_to_index[i][2] for i in list_to_index]) < (max_length - 20)
        # except:
        #     logging.warning("truncate too much")
        #     sys.exit(1)
        return list_to_index
