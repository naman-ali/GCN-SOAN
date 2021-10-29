from requires.Student import Student
import statistics
from requires.config import *


class Average:

    def __init__(self, experiment_id, df):

        self.experiment_id = experiment_id
        self.df = df
        self.class_size = len(self.df)
        self.number_of_graders = len(self.df.iloc[0]['graders'].split(','))

        self.run_model()
        self.save()

    def run_model(self):

        # print("average is running")

        for student_id, student in self.df.iterrows():

            student = Student(self.df, student_id)
            grades = list(student.get_grades())

            self.df.at[student_id, 'Average'] = round(statistics.mean(grades), 20)

        return self.df

    def save(self):

        experiment_file = STUDY_DIR + str(self.experiment_id) + '.csv'

        return self.df.to_csv(experiment_file, index=False)
