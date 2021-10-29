import statistics
from requires.Student import Student
from requires.config import *
import pandas as pd
from set_study import Fast_TunedModel

iterations = 300
get_last = -240
if Fast_TunedModel:
    iterations = 10
    get_last = -4


class TunedModel:

    def __init__(self, experiment_id, df):

        self.experiment_id = experiment_id
        self.df = self.get_df()
        self.class_size = len(self.df)
        self.number_of_graders = len(self.df.iloc[0]['graders'].split(','))
        self.su, self.tv, self.bv = {}, {}, {}
        self.b0_global = 0.1

        self.run_model()
        self.save()


    def get_df(self):

        experiment_file = STUDY_DIR + str(self.experiment_id) + '.csv'

        df = pd.read_csv(experiment_file)

        return df

    def run_model(self):

        # Generate an initial assignment to all non-observed variables, su, τv, bv for all true grades,
        # grader reliabilities and grader biases.
        for student_id, pd_student in self.df.iterrows():
            student_id = int(student_id)
            self.su[student_id], self.tv[student_id], self.bv[student_id] = [0.5], [10], [0]

        # start Gibbs iterations
        for t in range(0, iterations):

            print("TunedModel Iteration:", t)

            # For each user score sui (true score of user i)
            for student_id, pd_student in self.df.iterrows():

                student = Student(self.df, student_id)

                # hypers
                y0 = 5
                u0 = 0.1

                # mean
                sigma_0 = 0
                sigma_1 = 0
                for grader_id in student.get_graders():
                    grader_id = int(grader_id)

                    sigma_0 = sigma_0 + self.tv[grader_id][-1]
                    parentheses_0 = round(student.get_grade_by_grader(grader_id) + self.bv[int(grader_id)][-1], 20)
                    sigma_1 = round(sigma_1 + self.tv[grader_id][-1] * parentheses_0, 20)

                R = y0 + sigma_0
                mean = round(((y0 / R) * u0) + sigma_1 / R, 20)

                variance = y0 + sigma_0

                # add sui to su
                sui = round(np.random.normal(loc=mean, scale=1 / variance, size=1)[0], 20)
                sui = self.limit(sui, 0, 1)

                self.su[student_id].append(sui)

            # For each grader reliability sui (true score of user i)
            for student_id, pd_student in self.df.iterrows():

                student = Student(self.df, student_id)

                # hypers
                a0 = 2
                b0 = self.b0_global

                # mean
                shape = a0 + len(student.get_graders()) / 2

                # variance
                sigma_0 = 0
                for grading_id in student.get_gradings_students():
                    grading_id = int(grading_id)
                    grading_group = Student(self.df, grading_id)
                    zvu = round(grading_group.get_grade_by_grader(student_id), 20)
                    xvu = zvu - (self.su[grading_id][-1] + self.bv[student_id][-1])
                    sigma_0 = sigma_0 + pow(xvu, 2)


                rate = b0 + sigma_0 / 2


                # add tvi to tv
                tvi = np.random.gamma(shape=shape, scale=1/rate, size=1)[0]
                self.tv[student_id].append(tvi)

            # For each grader bias bvi (true score of user i)
            for student_id, pd_student in self.df.iterrows():

                student = Student(self.df, student_id)

                # hypers
                n0 = 0.1

                # variance
                sigma_0 = 0
                for grading_id in student.get_gradings_students():
                    grading_id = int(grading_id)
                    grading_group = Student(self.df, grading_id)
                    zvu = grading_group.get_grade_by_grader(student_id)
                    sigma_0 = sigma_0 + self.tv[student_id][-1] * (zvu - self.su[grading_id][-1])

                    # mean
                    mean = sigma_0 / (n0 + len(student.get_graders()) * self.tv[student_id][-1])

                    # variance
                    variance = n0 + len(student.get_graders()) * self.tv[student_id][-1]

                    # add bvi to bv
                    bvi = round(np.random.normal(loc=mean, scale=1 / variance, size=1)[0], 20)
                    bvi = self.limit(bvi, 0, 1)
                    self.bv[student_id].append(bvi)


        for index, each in self.su.items():
            self.su[index] = each[get_last:]

        return self.df

    def get_grader_bias(self, grader_id):
        return self.bv[int(grader_id)]

    def save(self):

        for index, each in self.su.items():
            self.df.at[index, 'TunedModel'] = round(statistics.mean(each), 20)


        experiment_file = STUDY_DIR + str(self.experiment_id) + '.csv'
        self.df.to_csv(experiment_file, index=False)

        return True

    def limit(self, sui, min, max):
        if sui < min:
            sui = min
        if sui > max:
            sui = max
        return sui


def run(experiment=None):

    TunedModel(experiment)
