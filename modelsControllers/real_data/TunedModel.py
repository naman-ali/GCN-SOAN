import numpy as np
import sys
from requires.helper import *
import statistics
from requires.GroupController import Group
from set_study import Fast_TunedModel

REAL_DATA_PROCESSED_DIR = ROOT_DIR + 'data/real_data_processed/'

iterations = 300
get_last = -240
if Fast_TunedModel:
    iterations = 3
    get_last = -1


class TunedModel:

    def __init__(self, df):

        self.df = df
        self.df.loc[self.df['grader_type'] == 'self', 'grader_type'] = 'peer'

        self.su, self.tv, self.bv = {}, {}, {}
        self.task_id = df['task_id'].unique().tolist()[0]

        self.run_model()

    def run_model(self):

        print("run model TunedModel")

        # Generate an initial assignment to all non-observed variables, su, τv, bv for all true grades,
        # grader reliabilities and grader biases.
        groups = self.df.groupby('group_id')
        for index, row in self.df.iterrows():
            self.su[row['group_id']], self.tv[row['group_id']], self.bv[row['group_id']] = [0.5], [10], [0]

        # start Gibbs iterations
        for t in range(0, iterations):

            print(" ----------- Iteration -----------", t)

            groups = self.df.groupby('group_id')

            # for each true score
            for index, row in self.df.iterrows():

                group = Group(self.df, row['group_id'])
                avg_group_grade = group.get_grades(grader_type='peer')['grade'].mean()
                ta_group_grade = group.get_grades(grader_type='TA')['grade'].mean()

                # hypers
                y0 = 5
                u0 = 0.1

                # mean
                sigma_0 = 0
                sigma_1 = 0
                for grader_id in group.get_graders():

                    grader_id = int(grader_id)
                    if grader_id not in self.tv.keys():
                        continue

                    sigma_0 = sigma_0 + self.tv[grader_id][-1]
                    parentheses_0 = group.get_grade_by_grader(grader_id) + self.bv[int(grader_id)][-1]
                    sigma_1 = sigma_1 + self.tv[grader_id][-1] * parentheses_0
                    # print(row['group_id'], grader_id, group.get_grade_by_grader(grader_id), self.bv[int(grader_id)][-1], parentheses_0)


                R = y0 + sigma_0
                mean = ((y0 / R) * u0) + sigma_1 / R
                # print(sigma_1)

                # variance
                variance = y0 + sigma_0

                # add sui to su
                sui = np.random.normal(loc=mean, scale=1/variance, size=1)[0]
                sui = limit(sui, 0, 1)

                self.su[row['group_id']].append(sui)
                # print("TA", ta_group_grade, "avg_grade", avg_group_grade, "sui:", sui, " mean:", mean, " variance:", variance)


            # For each grader reliability
            for index, row in self.df.iterrows():

                group = Group(self.df, row['group_id'])

                # hypers
                a0 = 1
                b0 = 0.1

                # mean
                shape = a0 + len(group.get_gradings()) / 2

                # variance
                sigma_0 = 0
                for grading_id in group.get_gradings_groups():

                    grading_id = int(grading_id)
                    grading_group = Group(self.df, grading_id)
                    zvu = grading_group.get_grade_by_grader(row['group_id'])
                    xvu = zvu - (self.su[grading_id][-1] + self.bv[row['group_id']][-1])
                    sigma_0 = sigma_0 + pow(xvu, 2)
                    # print(zvu, self.su[grading_id][-1], self.bv[row['group_id']][-1], xvu)


                rate = b0 + sigma_0/2
                # variance = 0.5

                # add tvi to tv
                tvi = np.random.gamma(shape=shape, scale=1/rate, size=1)[0]
                # tvi = random.randint(0, 150)
                self.tv[row['group_id']].append(tvi)
                # print("tvi:", tvi, "shape: ", shape, " scale:", 1/rate)
                # print('---')


            # For each grader bias
            for index, row in self.df.iterrows():

                group = Group(self.df, row['group_id'])

                # hypers
                n0 = 0.1

                # variance
                sigma_0 = 0
                for grading_id in group.get_gradings_groups():
                    grading_id = int(grading_id)
                    grading_group = Group(self.df, grading_id)
                    zvu = grading_group.get_grade_by_grader(row['group_id'])
                    # print("self.su[grading_id]:", self.su[grading_id])
                    sigma_0 = sigma_0 + self.tv[row['group_id']][-1]*(zvu - self.su[grading_id][-1])
                    # sigma_0 = 0

                # mean
                mean = sigma_0 / (n0 + len(group.get_graders())*self.tv[row['group_id']][-1])


                # variance
                variance = n0 + len(group.get_graders())*self.tv[row['group_id']][-1]

                # add bvi to bv
                bvi = np.random.normal(loc=mean, scale=1/variance, size=1)[0]
                bvi = limit(bvi, 0, 1)
                self.bv[row['group_id']].append(bvi)
                # self.bv[row['group_id']] = 0.02
                # print("bvi:", bvi, "mean:", mean, " variance:", variance)


        for index, each in self.su.items():
            self.su[index] = each[get_last:]

        self.save()

        return self.df


    def get_grader_bias(self, grader_id):
        return self.bv[int(grader_id)][-1]


    def save(self):

        models_df = pd.read_csv(REAL_DATA_PROCESSED_DIR + 'gg-models-results.csv')

        for group_id, each in self.su.items():
            row_index = models_df[(models_df['group_id'] == group_id) & (models_df['task_id'] == self.task_id)].index
            models_df.at[row_index, 'TunedModel_grade'] = statistics.mean(each)

        models_df.to_csv(REAL_DATA_PROCESSED_DIR + "gg-models-results.csv", index=False)

        return True
