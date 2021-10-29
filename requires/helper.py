import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
from requires.config import *
REAL_DATA_PROCESSED_DIR = ROOT_DIR + 'data/real_data_processed/'


def plot_distribution(data, experiment_id=None):

    x_min = 0.0
    x_max = 1.0

    # plot
    density = gaussian_kde(data)
    xs = np.arange(x_min, x_max, 0.01)
    density.covariance_factor = lambda: 0.4
    density._compute_covariance()
    plt.plot(xs, density(xs))

    # plot normal distribution with one peak
    mean = np.mean(data)
    std = np.std(data)

    count, bins, ignored = plt.hist(data, 20, density=True, color='lightgray')
    # plt.plot(bins, 1 / (std * np.sqrt(2 * np.pi)) *
    #          np.exp(- (bins - mean) ** 2 / (2 * std ** 2)),
    #          linewidth=2, color='r')


    plt.grid()

    plt.xlim(x_min, x_max)
    # plt.ylim(0, 0.25)

    plt.title('Distribution of grades - Class size: {}'.format(data.shape[0]), fontsize=10)

    plt.xlabel('x')
    plt.ylabel('Normal Distribution')

    plt.savefig("data/simulated_0.7dist/{}-actual-grades.png".format(experiment_id), dpi=300)
    plt.show()


def plot_graders_errors(df, experiment_id=None):


    df_array = []
    # for each student/grader
    for grader_index, row in df.iterrows():

        new_row = {}

        new_row['grader'] = grader_index
        new_row['grader_grade'] = row['actual_grade']

        errors = []
        # for each students he/she has graded
        gradings = row['gradings'].split(",")
        for grading in gradings:

            grading_df = df.loc[[int(grading)]]

            grading_graders = grading_df['graders'].item().split(",")
            grading_grades = grading_df['grades'].item().split(",")

            grader_position = grading_graders.index(str(grader_index))
            provided_grade = grading_grades[grader_position]

            error = round(abs(float(provided_grade) - grading_df['actual_grade'].item()), 2)
            errors.append(error)

        new_row['mean_errors'] = round(np.mean(np.array(errors)), 2)
        df_array.append(new_row)


    new_df = pd.DataFrame(df_array)

    new_df = new_df.sort_values(by='grader_grade')

    plt.scatter(range(0, new_df.index.shape[0]), new_df['mean_errors'].values, s=2)

    plt.title('The impact of students working ability on their grading ability.', fontsize=10)

    plt.xlabel('Student ids sorted by working ability grade (their own assignment grade)')
    plt.ylabel('Mean Absolute Error')

    plt.savefig("data/simulated_0.7dist/{}-working-on-grading.png".format(experiment_id), dpi=300, bbox_inchesstr='tight')

    plt.show()

    print(new_df)

    # sys.exit()


def plot_peer_grades(df, experiment_id=None):

    df = df.sort_values(by='actual_grade')

    df['student_id'] = range(df.index.shape[0])

    df_array = []
    for index, row in df.iterrows():

        grades = row['grades'].split(",")

        for grade in grades:

            new_row = {}
            new_row['student_id'] = row['student_id']
            new_row['actual_grade'] = row['actual_grade']
            new_row['peer_grade'] = grade

            df_array.append(new_row)

    new_df = pd.DataFrame(df_array)



    # new_df = new_df[new_df['student_id'] == 700]

    new_df['peer_grade'] = new_df['peer_grade'].astype(float)

    plt.scatter(new_df['student_id'].values, new_df['peer_grade'].values, s=0.9, c='orange')
    plt.scatter(new_df['student_id'].values, new_df['actual_grade'].values, s=0.2, c='black')


    plt.title('Comparison of peer grades and ground truth grades', fontsize=10)

    plt.xlabel('Student ids sorted by ground truth grade')
    plt.ylabel('Grade')

    plt.savefig("data/simulated_0.7dist/{}-peer-grades.png".format(experiment_id), dpi=300)

    plt.show()


def get_experiments_ids(study=None):
    experiments_info = pd.read_csv(EXPERIMENT_INFO_FILE)

    experiments_info = experiments_info[experiments_info['study'] == study]

    return list(experiments_info.experiment_id.values)


def how_many():
    how_many = 0
    for class_size in class_sizes:  # 3 items
        for number_of_grader in number_of_graders:  # 3 items
            for loc1 in locs1:  # 4 items ( mean from 10 to 40)
                for scale1 in scales1:  # 4 items ( scale from 5 to 20)
                    for graders_behaviour in graders_behaviours:
                        sizes1 = [x * int(class_size / 5) for x in range(0, 6)]
                        sizes1 = [sizes1[0]]
                        for size1 in sizes1:  # 6 items
                            for loc2 in locs2:  # 4 items ( mean from 60 to 90)
                                for scale2 in scales2:  # 4 items
                                    for control_grader_error in control_grader_errors:
                                        size2 = class_size - size1
                                        how_many = how_many + 1

    return how_many



def file_exist(path=None):
    return os.path.isfile(path)


def dir_exist(path=None):
    return os.path.isdir(path)


def make_dir(path=None):
    return os.makedirs(path)


def is_simulated_study():
    if STUDY_NUMBER in [1, 2, 3, 4, 5, 6, 9]:
        return True
    else:
        return False


def split_int(number):
    return [int(d) for d in str(number)]


# get maximum grade of each task
def get_max_grade(df_max, taskid):

    row = df_max[df_max['taskid'] == taskid]
    if row.empty:
        return "Task id does not exist"

    return row['maxgrade'].item()


def limit(sui, min, max):
    if sui < min:
        sui = min
    if sui > max:
        sui = max
    return sui


def is_graded_by_TA(labels=None, groupid=None, taskid=None):

    grade_df = labels[(labels['groupid'] == int(groupid)) &
                      (labels['taskid'] == int(taskid))]['label']


    grade = np.nan

    if not grade_df.empty:
        grade = grade_df.iloc[0]

    # # print(labels.sort_values(by='groupid'))
    # print(groupid, taskid, grade)
    # # print("----")
    # sys.exit()

    return grade


def implode(my_list):

    return ','.join(map(str, my_list))


def str_split(word):
    return [char for char in word]


def split_assignid(labels):

    for index, row in labels.iterrows():
        labels.loc[index, 'taskid'] = int(str(row['assignid']).split("-")[0])
        labels.loc[index, 'groupid'] = int(str(row['assignid']).split("-")[1])

    labels['taskid'] = labels['taskid'].astype('int')
    labels['groupid'] = labels['groupid'].astype('int')

    labels = labels.drop(columns=['assignid'])

    return labels


def get_labels(assignment, fold, set='train'):

    labels = pd.read_csv(REAL_DATA_PROCESSED_DIR + 'folds/assign{}-fold{}-{}.csv'.format(assignment, fold, set))
    labels = split_assignid(labels)

    return labels

