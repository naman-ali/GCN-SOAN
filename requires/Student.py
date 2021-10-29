
class Student:

    def __init__(self, df, student_id):
        """Initializes a user."""

        student_id = int(student_id)


        self.graders = list(map(int, df.iloc[student_id]['graders'].split(',')))
        self.grades = df.iloc[student_id]['grades'].split(',')
        self.grades = [float(i) for i in self.grades]
        # self.grades = [float(grade ) /100 for grade in self.grades]
        self.gradings = df.iloc[student_id]['gradings'].split(',')

        self.id = student_id
        self.grades = dict(zip(self.graders, self.grades))
        self.ta_grade = df.iloc[student_id]['actual_grade']

        self.get_similarity_instructor_sspa = 0

    def set_grades(self):
        self.grades = dict(zip(self.graders, self.grades))


    def get_grades_including_graders(self):
        return self.grades

    def get_graders(self):
        return self.grades.keys()

    def get_grades(self):
        return self.grades.values()

    def get_gradings_students(self):
        return self.gradings

    def get_gradings_grades(self):
        return self.grades.values()

    def get_grade_by_grader(self, grader_id):
        return float(self.grades[grader_id])


