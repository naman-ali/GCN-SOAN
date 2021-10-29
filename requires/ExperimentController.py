from data.DataController import Data

from modelsControllers.simulated_data.Average import Average
from modelsControllers.simulated_data.Median import Median
from modelsControllers.simulated_data.TunedModel import TunedModel
from modelsControllers.simulated_data.PeerRank import PeerRank
from modelsControllers.simulated_data.RankwithTA import run as RankwithTA_run
from modelsControllers.simulated_data.vancouver import run as Vancouver_run
from modelsControllers.simulated_data.gcn_soan import run as gcn_run
import pandas as pd
from requires.config import *
from requires import helper
import sys


class Experiment:

    info = dict()

    def __init__(self, data, models):

        self.data = data
        experiments_df = self.load_experiments_file()
        self.id = experiments_df.shape[0]

        self.models = models
        self.save_experiment_info()
        self.run()

    def run(self):

        # 1 - Data preparation and loader *************
        df = Data(self.id, self.data).get()

        # 2 - Train the Model *************************
        for model in self.models:
            print(model)
            # run model() ex peerrank()
            if model == 'GCN':
                gcn_run(self.id)
            elif model == 'RankwithTA':
                RankwithTA_run([self.id])
            elif model == 'Vancouver':
                Vancouver_run(self.id)
            else:
                getattr(sys.modules[__name__], model)(self.id, df)

        return True

    @staticmethod
    def load_experiments_file():

        if not helper.file_exist(EXPERIMENT_INFO_FILE):
            experiments_info = pd.DataFrame(columns=["experiment_id",
                                                     "study",
                                                     "run",
                                                     "class_size",
                                                     "number_of_graders",
                                                     "mean1",
                                                     "std1",
                                                     "size1",
                                                     "mean2",
                                                     "std2",
                                                     "size2",
                                                     "alpha",
                                                     "control_friendship",
                                                     "beta",
                                                     "Average_rmse",
                                                     "Median_rmse",
                                                     "PeerRank_rmse",
                                                     "GCN_rmse",
                                                     "RankwithTA_rmse",
                                                     "TunedModel_rmse",
                                                     "Vancouver_rmse"])
            experiments_info.to_csv(EXPERIMENT_INFO_FILE, index=False)
        else:
            experiments_info = pd.read_csv(EXPERIMENT_INFO_FILE)

        return experiments_info

    def save_experiment_info(self):

        experiments_info = pd.read_csv(EXPERIMENT_INFO_FILE)

        max_experiment_id = experiments_info['experiment_id'].max()

        if experiments_info.shape[0] == 0:
            max_experiment_id = 0

        self.info = {
            'experiment_id': experiments_info.shape[0],
            'study': self.data['study'],
            'run': self.data['run'],
            'class_size': self.data['class_size'],
            'number_of_graders': self.data['number_of_graders'],
            'mean1': self.data['actual_grades_distribution']['first'][0],
            'std1': self.data['actual_grades_distribution']['first'][1],
            'size1': self.data['actual_grades_distribution']['first'][2],
            'mean2': self.data['actual_grades_distribution']['second'][0],
            'std2': self.data['actual_grades_distribution']['second'][1],
            'size2': self.data['actual_grades_distribution']['second'][2],
            'alpha': self.data['graders_behaviours']['alpha'][0],
            'beta': self.data['control_grader_error'],
            'control_friendship': self.data['control_friendship']
        }

        info_df_row = pd.DataFrame([self.info])
        experiments_info = pd.concat([experiments_info, info_df_row])

        experiments_info.to_csv(EXPERIMENT_INFO_FILE, index=False)

