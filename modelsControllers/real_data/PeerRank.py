import numpy as np
import sys
import pandas as pd
from requires.helper import *

REAL_DATA_PROCESSED_DIR = ROOT_DIR + 'data/real_data_processed/'


class PeerRank:
    # constant attributes
    d = 0.7

    def __init__(self, df):

        self.df = df
        self.pre_process()

        self.class_size = len(self.df['group_id'].unique())
        self.task_id = df['task_id'].unique().tolist()[0]

        peer_ranks = self.run_model()
        self.save(peer_ranks)


    def pre_process(self):

        for name, group in self.df.groupby('task_id'):
            for each in range(1, 80):
                row = group[group['group_id'] == each]
                index = self.df.tail(1).index + 1
                if row.empty:
                    df2 = pd.DataFrame({'group_id': each,
                                        'task_id': name,
                                        'grader_group_id': 0,
                                        'grade': 0
                                        }, index=index)
                    self.df = self.df.append(df2)

        self.df.loc[self.df['grader_type'] == 'self', 'grader_type'] = 'peer'

        return self.df

    def run_model(self):

        # create a zero numpy array with class_size, class_size dimension
        G = np.zeros(shape=(self.class_size, self.class_size)).astype(float)

        for group_id in self.df['group_id'].unique().tolist():

            group = self.df[self.df['group_id'] == group_id]
            G[group_id-1, group_id-1] = 0
            graders = group['grader_group_id']
            grades = group['grade']
            grades_dict = dict(zip(graders, grades))

            for grader, grade in grades_dict.items():
                G[int(group_id-1), int(grader-1)] = grade


        G = np.transpose(G)
        row_count, col_count = G.shape
        G_copy = G.astype(float).copy()

        # G -> (Grader_group_id, Group_id)
        # G transpose -> (Group_id, Grader_group_id)
        G_transpose = np.transpose(G_copy)
        # average grade as initial ranks
        r = []
        for index, each in enumerate(G_transpose):
            group_id = index + 1
            number_of_graders = len(self.df[self.df['group_id'] == group_id]['grader_group_id'].unique())
            r.append(each.sum() / number_of_graders)

        previous_r = r.copy()

        for iteration in range(1, 50):

            # print(" ----- PeerRank itetation {} -------".format(iteration))

            for node_i in range(0, row_count):

                sigma = 0
                sum_graders_grades = 0

                for inlink in self.inlinks(node_i, G_copy):
                    # sigma = sigma + previous_r[inlink]
                    sigma = sigma + previous_r[inlink] * G_transpose[node_i, inlink]
                    sum_graders_grades = sum_graders_grades + previous_r[inlink]

                if sum_graders_grades != 0:
                    r[node_i] = (1 - self.d) * previous_r[node_i] + (self.d / sum_graders_grades) * sigma
                    r[node_i] = r[node_i]
                else:
                    r[node_i] = 0


            previous_r = r

        for index, each in enumerate(r):
            self.df.at[index, 'PeerRank'] = each

        return r

    def save(self, r):

        models_df = pd.read_csv(REAL_DATA_PROCESSED_DIR + 'gg-models-results.csv')

        for index, each in enumerate(r):
            row_index = models_df[(models_df['group_id'] == index+1) & (models_df['task_id'] == self.task_id)].index
            models_df.at[row_index, 'PeerRank_grade'] = each

        models_df.to_csv(REAL_DATA_PROCESSED_DIR + 'gg-models-results.csv', index=False)

        return True

    @staticmethod
    def outlinks(node, G):
        row_count, col_count = G.shape
        outlinks = []
        for col in range(0, col_count):
            if G[node, col] != 0:
                outlinks.append(col)

        return outlinks

    @staticmethod
    def inlinks(node, G):
        row_count, col_count = G.shape
        G = np.transpose(G)
        inlinks = []
        for col in range(0, col_count):
            if G[node, col] != 0:
                inlinks.append(col)

        return inlinks

