import numpy as np
import sys
import pandas as pd
from requires.helper import *
from sklearn.metrics import mean_squared_error
from requires import helper

REAL_DATA_PROCESSED_DIR = ROOT_DIR + 'data/real_data_processed/'


class RankwithTA:


    def __init__(self, df, fold=1, a=0.8, b=0.1, tune=False):

        # self.df = df[df['task_id'].isin(assignment_tasks)]
        self.a = a
        self.b = b
        self.df = df
        self.pre_process()
        self.fold = fold
        self.assignment = int(str_split(str(self.df['task_id'].unique()[0]))[0])
        self.train_labels = get_labels(self.assignment, fold, set='train')
        self.test_labels = get_labels(self.assignment, fold, set='test')
        self.class_size = len(self.df['group_id'].unique())
        self.task_id = df['task_id'].unique().tolist()[0]

        grades = self.run_model()
        self.save(grades)

    def pre_process(self):

        # give 0 to groups with no submission
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

        return self.df

    def run_model(self):

        # set initial grades
        prev_grades = dict()
        for group_id, group in self.df.groupby('group_id'):
            prev_grades[group_id] = group.grade.mean()


        grades = prev_grades.copy()

        for iteration in range(1, 5):

            # print(" ----- RankwithTA itetation {} -------".format(iteration))

            grade_by_TA_count = 0
            # for each student/group i
            for group_id, group in self.df.groupby('group_id'):

                # for each grade given to student i
                sigma_numerator = 0
                sigma_denominator = 0
                for index, row in group.iterrows():

                    taskid = self.df['task_id'].unique()[0]
                    grade = row['grade']

                    # get grader grade  TA | previous predicted grade
                    if row['grader_group_id'] != 0:
                        grader_grade = prev_grades[row['grader_group_id']]
                        grader_TA_grade = is_graded_by_TA(labels=self.train_labels, groupid=row['grader_group_id'], taskid=taskid)
                        if not np.isnan(grader_TA_grade):
                            grader_grade = grader_TA_grade
                            grade_by_TA_count = grade_by_TA_count + 1
                    else:
                        grader_grade = 1

                    sigma_numerator = sigma_numerator + (grade*grader_grade)
                    sigma_denominator = sigma_denominator + grader_grade

                new_grade = (1-self.a-self.b)*prev_grades[group_id] + self.a*(sigma_numerator/sigma_denominator)

                grades[group_id] = new_grade

            prev_grades = grades.copy()

        return grades


    def save(self, grades):

        if not helper.file_exist(REAL_DATA_PROCESSED_DIR + '/RankwithTA.csv'):
            df = pd.DataFrame(columns=['group_id', 'taskid', 'fold', 'grade', 'ta_grade', 'test_set'])
            df.to_csv(REAL_DATA_PROCESSED_DIR + '/RankwithTA.csv', index=False)

        results = pd.read_csv(REAL_DATA_PROCESSED_DIR + '/RankwithTA.csv')

        for group_id, grade in grades.items():

            exist = results[(results['group_id'] == group_id) &
                            (results['taskid'] == self.task_id) &
                            (results['fold'] == self.fold)]

            if exist.empty:

                isin_test = self.test_labels[(self.test_labels['taskid'] == self.task_id) &
                                             (self.test_labels['groupid'] == group_id)]

                if isin_test.empty:
                    isin_test_set = 0
                else:
                    isin_test_set = 1

                # get TA grade
                TA_grade = is_graded_by_TA(labels=self.test_labels, groupid=group_id, taskid=self.task_id)

                row = {
                    "group_id": group_id,
                    "taskid": self.task_id,
                    "fold": self.fold,
                    "grade": grade,
                    "ta_grade": TA_grade,
                    "test_set": isin_test_set
                }

                results.loc[results.size] = row.values()

            else:
                results.loc[exist.index[0], 'grade'] = grade


        results = results.dropna()

        # group_id, taskid, fold, grade, ta_grade, test_set, assgin_id
        results['group_id'] = results['group_id'].astype(int)
        results['taskid'] = results['taskid'].astype(int)
        results['fold'] = results['fold'].astype(int)
        results['test_set'] = results['test_set'].astype(int)
        results['grade'] = results['grade'].astype(float)
        results['ta_grade'] = results['ta_grade'].astype(float)

        results.to_csv(REAL_DATA_PROCESSED_DIR + 'RankwithTA.csv', index=False)

        return True


