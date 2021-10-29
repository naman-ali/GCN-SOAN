from scipy import stats
from requires.helper import *
from requires.config import *
import networkx as nx

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


class Data:

    graders_erros = []

    index_col = 'id'

    def __init__(self, experiment_id, data):

        self.experiment_id = experiment_id
        self.data = data
        self.max_grade = data['max_grade']

        # Instead of "generate", it can be a file name, then it will not generate data and only loads that file
        if data['source'] == 'generate':
            self.class_size = self.data['class_size']
            self.number_of_graders = self.data['number_of_graders']
            self.actual_grades_distribution = self.data['actual_grades_distribution']

            self.df = self.generate()

        else:
            # csv data to data-frame
            self.df = self.load()

    def load(self):
        return pd.read_csv(self.processed_directory + self.data['source'], index_col=self.index_col)

    def get(self):
        return self.df

    # start generating data
    def generate(self):

        # generate students actual grades (Ground truth - TA)
        self.df = self.generate_actual_grades()

        # plot_distribution(self.df.actual_grade.values, experiment_id=self.experiment_id)

        self.df = self.assign_graders()

        self.df = self.assign_gradings()

        self.df = self.graders_behaviours()

        self.df = self.generate_friendships()

        self.df = self.generate_grades_by_graders()

        self.df = self.assign_gradings_grades()

        self.graders_erros = np.array(self.graders_erros)

        # save the final generated data in a csv file
        experiment_file = STUDY_DIR + str(self.experiment_id) + '.csv'
        self.df.to_csv(experiment_file, index=False)

        return self.df

    # generate actual grades
    def generate_actual_grades(self):

        mean1, variance1, size1 = self.actual_grades_distribution['first']
        mean2, variance2, size2 = self.actual_grades_distribution['second']
        grades = np.concatenate([np.random.normal(loc=mean1, scale=variance1, size=size1),
                                 np.random.normal(loc=mean2, scale=variance2, size=size2)])

        # non-multimodal distribution
        # loc3, scale3, size3 = (50, 10, self.max_grade)
        grades = np.round(grades, 20)
        np.random.shuffle(grades)

        index = 0
        for grade in grades:
            if grade > self.max_grade:
                grades[index] = self.max_grade
            elif grade < 0:
                grades[index] = 0
            index = index + 1

        # show distribution plot
        # self.plotdist(grades)

        self.df = pd.DataFrame({'actual_grade': grades})

        return self.df

        # save to file
        # with open('{}data-{}.csv'.format(self.processed_directory, self.experiment_id), 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['id', 'actual_grade'])
        #     for index, val in enumerate(grades):
        #         if val > self.max_grade:
        #             val = self.max_grade
        #         elif val < 0:
        #             val = 0
        #         writer.writerow([index, val])
        #
        # return pd.read_csv('{}data-{}.csv'.format(self.processed_directory, self.experiment_id), index_col='id')

    # generate grades by graders
    def assign_graders(self):

        # add student grades for each student
        all_assigned_graders = {}
        potential_graders = self.df.index.copy().tolist()

        # for each student (each row of data.csv file)
        for index, row in self.df.iterrows():
            # choose grader
            graders = []
            while_count = 0
            while len(graders) < self.number_of_graders:
                while_count = while_count + 1

                if while_count > self.number_of_graders * 7:
                    grader_id = np.random.choice(range(1, len(self.df)))
                else:
                    grader_id = np.random.choice(potential_graders)

                # check if the grader is not grading his own submission
                if grader_id not in graders and grader_id != index:
                    graders.append(grader_id)
                    if grader_id not in all_assigned_graders.keys():
                        all_assigned_graders[grader_id] = 0
                    all_assigned_graders[grader_id] = all_assigned_graders[grader_id] + 1
                    if all_assigned_graders[grader_id] == self.number_of_graders:
                        potential_graders.remove(grader_id)

            # add his graders ids (graders column)
            graders_string = ','.join([str(i) for i in graders])
            self.df.at[index, 'graders'] = graders_string

        return self.df

    def assign_gradings(self):

        # for each student (each row of data.csv file)
        gradings = {}
        for student_id, student in self.df.iterrows():
            graders = student['graders'].split(',')
            for grader_id in graders:
                if grader_id not in gradings:
                    gradings[grader_id] = []
                gradings[grader_id].append(student_id)

        for key, grading in gradings.items():
            gradings_string = ','.join([str(i) for i in grading])
            self.df.at[int(key), 'gradings'] = gradings_string

        return self.df

    def assign_gradings_grades(self):

        for student_id, student in self.df.iterrows():
            gradings = student['gradings'].split(',')
            gradings_grades = []
            for grading in gradings:
                peer_grade = self.get_peer_grade(student_id=grading, grader_id=student_id)
                gradings_grades.append(peer_grade)

            self.df.loc[student_id, 'gradings_grades'] = graders_string = ','.join([str(i) for i in gradings_grades])

        return self.df

    def get_peer_grade(self, student_id, grader_id):


        student_df = self.df.loc[[int(student_id)]]

        grader_index = student_df['graders'].iloc[0].split(',').index(str(grader_id))

        peer_grade = student_df['grades'].iloc[0].split(',')[grader_index]

        return float(peer_grade)

    def graders_behaviours(self):

        for index, row in self.df.iterrows():
            alpha = np.random.choice(self.data['graders_behaviours']['alpha'], 1, p=self.data['graders_behaviours']['probabilties'])
            self.df.at[index, 'grader_behaviour'] = alpha

        return self.df

    def generate_friendships(self):

        if self.data['friendship_type'] == 'none':
            self.df['friends'] = ''

            return self.df

        if self.data['friendship_type'] == 'erdos':
            n = self.df.shape[0]
            p = self.data['control_friendship']
            G = nx.erdos_renyi_graph(n, p, seed=None, directed=False)

            for i in range(0, n):
                neighbours = list(dict(G[i]).keys())
                neighbours_str = ','.join([str(i) for i in neighbours])
                self.df.loc[i, 'friends'] = neighbours_str

        if self.data['friendship_type'] == 'homophily':
            for index, row in self.df.iterrows():
                grade = row['actual_grade']
                friends = self.df[(self.df['actual_grade'] <= grade + self.data['control_friendship']) &
                                  (self.df['actual_grade'] >= grade - self.data['control_friendship'])]

                if not friends.empty:
                    friends_list = list(friends.index)
                    friends_list.remove(index)
                    friends_str = ','.join([str(i) for i in friends_list])

                self.df.loc[index, 'friends'] = friends_str

        return self.df

    # for each student call generate_grade_by_grader method and store the result in self.df
    def generate_grades_by_graders(self):

        for index, row in self.df.iterrows():

            graders = list(map(int, row['graders'].split(",")))

            # add his/her grades by graders to (grades column)
            grades = ""
            for grader in graders:

                is_friend = False
                if row['friends'] != '':
                    friends = list(map(int, row['friends'].split(",")))
                    if grader in friends:
                        is_friend = True

                grade = self.generate_grade_by_grader(row['actual_grade'], grader, is_friend=is_friend)
                grades = grades + "," + str(grade)

            self.df.at[index, 'grades'] = grades.strip(",")

        return self.df

    # generate Mij ~ N(Xj, 6i)
    def generate_grade_by_grader(self, student_grade, grader, is_friend=False):

        # student_grade = 50
        # grader_grade = self.max_grade
        if is_friend and self.data['friendship_type'] == 'erdos':
            return self.max_grade

        grader_grade = self.df.iloc[grader]['actual_grade']

        grader_error_scale = self.grader_error(grader_grade)

        self.graders_erros.append(grader_error_scale)

        alpha = self.df.iloc[grader]['grader_behaviour']

        mean1, variance1, size1 = (student_grade, grader_error_scale, 1)
        # grade = np.random.normal(loc=mean1, scale=variance1, size=size1)
        # mean1 = mean1 - alpha

        if self.data['skewness_method'] == 'azzalini':
            grade = self.randn_skew_fast(size1, alpha=alpha, loc=mean1, scale=grader_error_scale)
        elif self.data['skewness_method'] == 'biased':
            grade = np.random.normal(loc=mean1+alpha, scale=grader_error_scale, size=size1)

        grade = round(grade[0], 20)

        if grade > self.max_grade:
            grade = self.max_grade
        elif grade < 0:
            grade = 0

        return round(grade, 20)

    # 6i = N(h(Xi), 6^i)
    def grader_error(self, grader_grade):

        if self.data['grader_error_type'] == 'constant_std':
            grader_error = self.data['control_grader_error']

        elif self.data['grader_error_type'] == 'working_impact_grading_bt':

            # monotonic function
            grader_error = 0.25*(self.max_grade - self.data['control_grader_error']*grader_grade)


            if grader_error < 0:
                grader_error = 0
            elif grader_error > self.max_grade:
                grader_error = self.max_grade

        elif self.data['grader_error_type'] == 'working_impact_grading':

            # monotonic function
            # if grader_grade != 0:
                # grader_error = self.data['control_grader_error'] / grader_grade * self.max_grade
            grader_error = self.data['control_grader_error'] * (self.max_grade - grader_grade)
            # else:
            #     grader_error = self.max_grade/2

            if grader_error < 0:
                grader_error = 0
            elif grader_error > self.max_grade:
                grader_error = self.max_grade

        return round(grader_error, 20)
