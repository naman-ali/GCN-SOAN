import pandas as pd
from sklearn.metrics import mean_squared_error

from requires.config import *


class RankwithTA:

    def __init__(self, experiment_id, fold=1):

        self.experiment_id = experiment_id
        self.fold = fold
        self.df = self.get_df()

        self.train = self.get_set(set='train')
        self.test = self.get_set(set='test')
        # self.test_ids = get_set(self.experiment_id, fold, set='test')
        self.class_size = self.df.shape[0]
        self.max_grade = 1

        grades = self.run_model()
        self.save(grades)

    def get_df(self):
        experiment_file = STUDY_DIR + str(self.experiment_id) + '.csv'

        df = pd.read_csv(experiment_file)

        return df

    def run_model(self):

        # set initial grades
        prev_grades = dict()
        for student_id, student in self.df.iterrows():
            student_grades = np.array(student['grades'].split(',')).astype('float')
            prev_grades[student_id] = round(np.mean(student_grades), 20)


        grades = prev_grades.copy()
        grades_backup = prev_grades.copy()

        # algorithm main iteration
        for iteration in range(1, 5):

            print(" ----- itetation {} -------".format(iteration))

            grade_by_TA_count = 0
            # for each student/group i
            for student_id, student in self.df.iterrows():

                student_grades = np.array(student['grades'].split(',')).astype('float')
                graders = np.array(student['graders'].split(',')).astype('int')
                gradings = np.array(student['gradings'].split(',')).astype('int')
                gradings_grades = np.array(student['gradings_grades'].split(',')).astype('float')

                # for each grade given to student i
                sigma_numerator = 0
                sigma_denominator = 0
                for index, grade in enumerate(student_grades):

                    # get grader grade TA | previous predicted grade
                    grader_id = graders[index]
                    grader_grade = prev_grades[grader_id]
                    grader_TA_grade = self.is_graded_by_TA(student_id=grader_id)
                    # if not np.isnan(grader_TA_grade):
                    if grader_TA_grade != 404:
                        grader_grade = grader_TA_grade
                        grade_by_TA_count = grade_by_TA_count + 1

                    sigma_numerator = round(sigma_numerator + (grade*grader_grade), 20)
                    sigma_denominator = round(sigma_denominator + grader_grade, 20)
                    if sigma_denominator == 0:
                        sigma_denominator = 0.001


                # for each grade given by student i
                sigma2_numerator = 0
                sigma2_denominator = 0
                for index, grading in enumerate(gradings):

                    peer_grade = gradings_grades[index]

                    best_available_grade = prev_grades[grading]
                    TA_grade = self.is_graded_by_TA(student_id=grading)
                    if TA_grade != 404:
                        best_available_grade = TA_grade

                    sigma2_numerator = round(sigma2_numerator + self.max_grade - round(abs(peer_grade - best_available_grade), 2), 20)
                    sigma2_denominator = len(gradings)

                new_grade = 0.1*prev_grades[student_id] + 0.8*(sigma_numerator/sigma_denominator) + 0.1*(sigma2_numerator/sigma2_denominator)

                grades[student_id] = round(abs(new_grade), 20)


            prev_grades = grades.copy()


        return grades

    def save(self, grades):

        results = self.df

        fold_column_gcn = "gcn_fold" + str(self.fold)

        fold_student_ids = self.df[self.df[fold_column_gcn].notnull()].index

        grades = np.array(list(grades.values()))

        fold_column_RankwithTA = "RankwithTA_fold" + str(self.fold)
        results[fold_column_RankwithTA] = np.nan
        results.loc[fold_student_ids, fold_column_RankwithTA] = grades[fold_student_ids]

        fold_results = results[results[fold_column_RankwithTA].notnull()]

        rmse_avg = round(mean_squared_error(fold_results['actual_grade'], fold_results['Average'], squared=False), 20)
        rmse_RankwithTA = round(mean_squared_error(fold_results['actual_grade'], fold_results[fold_column_RankwithTA], squared=False), 20)
        rmse_gcn = round(mean_squared_error(fold_results['actual_grade'], fold_results['gcn_fold'+str(self.fold)], squared=False), 20)

        print("fold:", self.fold,  "rmse_avg:", rmse_avg, "rmse_RankwithTA:", rmse_RankwithTA, "rmse_gcn:", rmse_gcn)

        experiment_file = STUDY_DIR + str(self.experiment_id) + '.csv'
        results.to_csv(experiment_file, index=False)

        return True


    def str_split(self, word):
        return [char for char in word]

    def split_assignid(self, labels):

        for index, row in labels.iterrows():
            labels.loc[index, 'taskid'] = int(str(row['assignid']).split("-")[0])
            labels.loc[index, 'groupid'] = int(str(row['assignid']).split("-")[1])

        labels['taskid'] = labels['taskid'].astype('int')
        labels['groupid'] = labels['groupid'].astype('int')

        labels = labels.drop(columns=['assignid'])

        return labels

    def get_set(self, set='train'):

        fold_column_gcn = "gcn_fold"+str(self.fold)
        if set == 'train':
            df_set = self.df[self.df[fold_column_gcn].isnull()]
        if set == 'test':
            df_set = self.df[self.df[fold_column_gcn].notnull()]

        return df_set

    def is_graded_by_TA(self, student_id=None):

        if student_id in self.train.index:
            grade = self.train.loc[[student_id]]['actual_grade'].iloc[0]
            grade = abs(grade)
        else:
            grade = 404

        return grade

    def get_peer_grade(self, student_id, grader_id):

        student_df = self.df.loc[[student_id]]

        grader_index = student_df['graders'].iloc[0].split(',').index(str(grader_id))

        peer_grade = student_df['grades'].iloc[0].split(',')[grader_index]

        return float(peer_grade)


def run(experiments=None):

    folds = [1, 2, 3, 4]

    for experiment in experiments:
        for fold in folds:
            RankwithTA(experiment, fold=fold)
