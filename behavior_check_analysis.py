from definition import MODEL_ROOT, PROJECT_ROOT, GLOBAL_INDEX
from utils import json_load, output_history
from statsmodels.stats.weightstats import ztest
from scipy import stats

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import sys
import csv
import codecs
import logging
logging.getLogger().setLevel(logging.INFO)

class Analysis(object):
    def __init__(self,wordpair,model_path,setting):
        self.wordpair = wordpair
        self.model_path = os.path.join(MODEL_ROOT,model_path)
        if setting == 'word_choice':
            self.accuracy_type = "wordpair_accuracy"
        else:
            self.accuracy_type = "eval_accuracy"
        
        self.key = model_path.split('_')[1]
        logging.info("model_path:{}".format(self.model_path))

    def load_behavior_score(self):
        
        normal_result = json_load(self.model_path,"behavior_normal_eval_result_v3.json")
        reverse_result = json_load(self.model_path,"behavior_reverse_eval_result_v3.json") 
        double_reverse_result = json_load(self.model_path,"behavior_double_reverse_eval_result_v3.json")

        return normal_result,reverse_result,double_reverse_result

    def compute_t_score(self):
        
        normal_result, reverse_result, double_reverse_result = self.load_behavior_score()
        
        x = len(normal_result[self.accuracy_type])
        print("normal average accuracy:\t{}\n".format(sum(normal_result[self.accuracy_type])/x))
        print("reverse average accuracy:\t{}\n".format(sum(reverse_result[self.accuracy_type])/x))    
        print("double reverse average accuracy:\t{}\n".format(sum(double_reverse_result[self.accuracy_type])/x))

        t_score, p_value = stats.ttest_rel(normal_result[self.accuracy_type],reverse_result[self.accuracy_type])

        print("t_score:{}\t p_value:{}".format(t_score,p_value))
        analysis_result = {}
        analysis_result['t_score'] = t_score
        analysis_result['p_value'] = p_value

        output_history(self.output_path,analysis_result)
        
    def compute_z_score(self):
        
        normal_result, reverse_result, double_reverse_result = self.load_behavior_score()
        
        x = len(normal_result[self.accuracy_type])
        print("normal average accuracy:\t{}\n".format(sum(normal_result[self.accuracy_type])/x))
        print("reverse average accuracy:\t{}\n".format(sum(reverse_result[self.accuracy_type])/x))    
        print("double reverse average accuracy:\t{}\n".format(sum(double_reverse_result[self.accuracy_type])/x))

        z_score, p_value = ztest(normal_result[self.accuracy_type],
                                                         reverse_result[self.accuracy_type])

        print("test_stats:{}\t p_value:{}".format(z_score,p_value))
        analysis_result = {}
        analysis_result['test_stats'] = z_score
        analysis_result['p_value'] = p_value

        output_history(self.output_path, analysis_result)
        
    def calculate_average(self, wordpair):
        normal_result, reverse_result, double_reverse_result = self.load_behavior_score()
        x = len(normal_result[self.accuracy_type])
        
        outfile = codecs.open(os.path.join(MODEL_ROOT,'behavior_full_result_0526.csv'),'a+',encoding='utf-8')
        outfile_writer = csv.writer(outfile, delimiter=',')
        outfile_writer.writerow(["|".join(wordpair),
                                 sum(normal_result[self.accuracy_type])/x,
                                 sum(reverse_result[self.accuracy_type])/x,
                                 sum(double_reverse_result[self.accuracy_type])/x,
                                 (sum(normal_result[self.accuracy_type])/x) - (sum(reverse_result[self.accuracy_type])/x)
                                ])
    def compute_overall_t_score(self):
        file_path = os.path.join(MODEL_ROOT, 'behavior_full_result_0526.csv')
        with codecs.open(file_path, 'r', encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile,  delimiter=',')
            normal = []
            reverse = []
            for row in csv_reader:
                normal.append(float(row[1]))
                reverse.append(float(row[2]))
        difference = sum(normal)/len(normal) - sum(reverse)/len(reverse)
        
        
        t_score, p_value = stats.ttest_rel(normal,reverse)

        print("t_score:{}\t p_value:{}".format(t_score,p_value))
        analysis_result = {}
        analysis_result['t_score'] = t_score
        analysis_result['p_value'] = p_value
        analysis_result['difference'] = difference
        file_path = os.path.join(MODEL_ROOT,'behavior_full_t_test_0526.json')
        with codecs.open(file_path,'w',encoding="utf-8") as outfile:
            json.dump(analysis_result,outfile,indent=4)
            
    def draw_behavior_check_figure(self):
        normal_result, reverse_result, double_reverse_result = self.load_behavior_score()
        
        linewidth = 1.5
        fig, ax = plt.subplots(figsize=(12, 4.8))

        graph_path = os.path.join(PROJECT_ROOT,"EMNLP_figure",
                                  "ENT_regular_{:0>2}_{}_behavior_check_{}.png".format(
                                      self.key, 
                                      "_".join(sorted(self.wordpair)),
                                      self.accuracy_type
                                      )
                                 )

        x = [ i for i, _ in enumerate(normal_result[self.accuracy_type])]

        for setting in ['normal','counter']:
            if setting == 'normal':
                y = normal_result[self.accuracy_type]
            elif setting == 'counter':
                y = reverse_result[self.accuracy_type]
            ax.plot(x, y, label="{}_acc".format(setting), linewidth=linewidth)

        ax.legend(loc="upper left")
        ax.set_ylabel('accuracy',fontsize=10)
        ax.set_xlabel('batch',fontsize = 10)
        #ax.set(title="{} no counter behavior check".format(" ".join(self.wordpair))) 
        fig.savefig(graph_path, dpi=300)

if __name__ == '__main__':
    # ==== single model case ==== #
    #wordpair = sys.argv[1:3]
    #model_dir = sys.argv[3]
    #setting = sys.argv[4]
    #draw = Analysis(wordpair,model_dir,setting)
    #draw.calculate_average(wordpair)
    #draw.compute_overall_t_score()
    #draw.compute_z_score()
    #draw.draw_behavior_check_figure()
    
    # ==== full model case ==== #
    normal_y = []
    reverse_y = []
    for wordpair_str, key in GLOBAL_INDEX.items():
        wordpair = wordpair_str.split('|')
        key = int(key)
        setting = sys.argv[4]
        model_dir = "model_{}_entailment".format(key)
        #model_dir = "model_{}_FITB".format(key)
        draw = Analysis(wordpair,model_dir,setting)
        normal_result,reverse_result,double_reverse_result = draw.load_behavior_score()
        normal_y.extend(normal_result[draw.accuracy_type])
        reverse_y.extend(reverse_result[draw.accuracy_type])
        
    linewidth = 0.5
    fig, ax = plt.subplots(figsize=(12, 4.8))

    graph_path = os.path.join(PROJECT_ROOT,"EMNLP_figure",
                                  "ENT_NC_behavior_check_0604_v2.png"
                                 )

    x = [ i for i, _ in enumerate(normal_y)]

    for setting in ['Good','Bad']:
        if setting == 'Good':
            y = normal_y
        elif setting == 'Bad':
            y = reverse_y
        ax.plot(x, y, label="{} materials".format(setting), linewidth=linewidth)

    ax.legend(loc="upper left",fontsize=16)
    ax.set_ylabel('Accuracy',fontsize=16)
    ax.set_xlabel('Quiz ID',fontsize = 16)
    #ax.set(title="Entailment no counter model whole behavior check") 
    fig.savefig(graph_path, dpi=300,bbox_inches = 'tight',
    pad_inches = 0)