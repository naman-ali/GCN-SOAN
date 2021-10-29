from requires.ExperimentController import Experiment
from requires.interpret_results import run as interpret_results_run
from requires.helper import *
from requires.homophily import run as homophily_run


if __name__ == '__main__':

    # studies on simulated data
    if is_simulated_study():
        if study != 'homophily':

            for run in range(1, runs+1):
                # generate data
                generated_count = 0
                processes = []
                for class_size in class_sizes:
                    for number_of_grader in number_of_graders:
                        for loc1 in locs1:
                            for graders_behaviour in graders_behaviours:
                                for control_friendship in control_friendships:
                                    for scale1 in scales1:
                                        sizes1 = [x * int(class_size/5) for x in range(0, 6)]
                                        sizes1 = [sizes1[left_pick_portion]]
                                        for size1 in sizes1:
                                            for loc2 in locs2:
                                                for scale2 in scales2:
                                                    for control_grader_error in control_grader_errors:
                                                        size2 = class_size - size1

                                                        # Experiment
                                                        data = {
                                                            'source': 'generate',
                                                            'class_size': class_size,
                                                            'number_of_graders': number_of_grader,
                                                            'actual_grades_distribution': {
                                                                'first': (loc1, scale1, size1),
                                                                'second': (loc2, scale2, size2)
                                                            },
                                                            'graders_behaviours': graders_behaviour,
                                                            'control_grader_error': control_grader_error,
                                                            'max_grade': max_grade,
                                                            'grader_error_type': grader_error_type,
                                                            'skewness_method': skewness_method,
                                                            'study': study_title,
                                                            'run': run,
                                                            'friendship_type': friendship_type,
                                                            'control_friendship': control_friendship
                                                        }

                                                        Experiment(data, MODELS)

                                                        generated_count = generated_count + 1

            experiments = get_experiments_ids(study=study_title)

            interpret_results_run(study=study_title, experiments=experiments, x_axis=x_axis)


        if study == 'homophily':
            homophily_run()

    # studies on real data
    if not is_simulated_study():
        from requires import real_data

        real_data.run()

