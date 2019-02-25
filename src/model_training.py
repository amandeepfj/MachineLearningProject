# Created by Amandeep at 2/25/2019
# "We are drowning in information, while starving for wisdom - E. O. Wilson"

# Read data from sqllite database which has all the csv files dumped earlier.
import sqlite3
import pandas as pd
import numpy as np
from surprise import Reader
from surprise import KNNBaseline
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset
import pickle

conn = sqlite3.connect("C:\Users\Amandeep\pluralsight.db")

user_assessment_scores = pd.read_sql_query("select * from user_assessment_scores limit 1000;", conn)
user_course_views = pd.read_sql_query("select * from user_course_views limit 1000;", conn)
course_tags = pd.read_sql_query("select * from course_tags limit 1000;", conn)
user_interests = pd.read_sql_query("select * from user_interests limit 1000;", conn)

# Close connection of sqlite. Always remember to close the connection!
conn.close()

# Lets update the datatypes.
user_assessment_scores['user_assessment_score'] = pd.to_numeric(user_assessment_scores['user_assessment_score'])
user_assessment_scores['user_assessment_date'] = pd.to_datetime(user_assessment_scores['user_assessment_score'])
user_course_views['view_time_seconds'] = pd.to_numeric(user_course_views['view_time_seconds'])
user_course_views['view_date'] = pd.to_datetime(user_course_views['view_date'])
user_interests['date_followed'] = pd.to_datetime(user_interests['date_followed'])

if user_assessment_scores.drop_duplicates().shape[0] != user_assessment_scores.shape[0]:
    user_assessment_scores = user_assessment_scores.drop_duplicates()

if user_course_views.drop_duplicates().shape[0] != user_course_views.shape[0]:
    user_course_views = user_course_views.drop_duplicates()

if course_tags.drop_duplicates().shape[0] != course_tags.shape[0]:
    course_tags = course_tags.drop_duplicates()

if user_interests.drop_duplicates().shape[0] != user_interests.shape[0]:
    user_interests = user_interests.drop_duplicates()

OPTIMIZED_DIM_SIZE_USER_ASSESSMENT = 25
OPTIMIZED_DIM_SIZE_USER_INTEREST = 40
OPTIMIZED_DIM_SIZE_USER_COURSE_TAG = 80
OPTIMIZED_DIM_SIZE_USER_COURSE = 100
OPTIMIZED_DIM_SIZE_USER_COURSE_LEVEL = 3

TOP_CUTOFF_USERS = 10


# Class for similarity measure.
class SimilarityMeasure:
    def __initialize_matrix(self):
        if self.__values_column is None:
            self.__dataframe['temp_val'] = 1
            self.__values_column = 'temp_val'
        self.__scores_matrix = self.__dataframe.pivot(index=self.__index_column, columns=self.__columns_column,
                                                      values=self.__values_column)
        self.__scores_matrix = self.__scores_matrix.fillna(0)
        self.index_values = self.__scores_matrix.index.values
        # we need reindexed to calclulate the cosine similarities
        self.__reindexed_scores_matrix = self.__scores_matrix.copy()
        self.__reindexed_scores_matrix.index = range(0, self.index_values.shape[0])
        self.pearson_similarity_martix = None
        self.cosine_similarities_matrix = None
        self.__sliced = None
        self.__knn_algo = None

    def calculate_pearson_similarity(self):
        self.pearson_similarity_martix = self.__scores_matrix.T.corr(method='pearson')
        print("Pearson similarity calculated!")

    # Function that taken in user handle as input and outputs most similar users based on pearson.
    def get_pearson_similar_users(self, user_handle, num_similar_users=TOP_CUTOFF_USERS):
        if not self.isValidUser(user_handle):
            print("Error - User not found!!")
            return None
        if self.pearson_similarity_martix is not None:
            user_handle_scores = pd.DataFrame(self.pearson_similarity_martix[user_handle])
            user_handle_scores.columns = [0]
            similar_users = user_handle_scores.sort_values(by=[0], ascending=False)[1:num_similar_users]
            # Normalize between 0 to 1
            similar_users[0] = (similar_users[0] - min(similar_users[0])) / (
                    max(similar_users[0]) - min(similar_users[0]))
            return similar_users
        else:
            print("Error - Pearson similarity not calculated!!!!")

    # Function to calculate cosine similarity.
    def calculate_cosine_similarity(self):
        A_sparse = sparse.csr_matrix(self.__reindexed_scores_matrix)
        self.cosine_similarities_matrix = cosine_similarity(A_sparse, dense_output=False)
        print("Cosine similarity calculated!")

    # Function that taken in user handle as input and outputs most similar users based on pearson.
    def get_cosine_similar_users(self, user_handle, num_similar_users=TOP_CUTOFF_USERS):
        if not self.isValidUser(user_handle):
            print("Error - User not found!!")
            return None
        if self.cosine_similarities_matrix is not None:
            idx = np.where(self.index_values == user_handle)
            scores = pd.DataFrame(self.cosine_similarities_matrix[idx].T.toarray(), index=self.index_values)
            similar_users = scores.sort_values(by=[0], ascending=False)[1:num_similar_users]
            # Normalize between 0 to 1
            # similar_users[0] = (similar_users[0] - min(similar_users[0])) / (max(similar_users[0]) - min(similar_users[0]))
            # similar_users = similar_users.fillna(0)
            return similar_users
        else:
            print("Error - Cosine similarity not caclculated!!!!")

    def calculate_svd_similarity(self, full_matrix=False, dim_size=20):
        scores_mean = np.asarray([(np.mean(self.__reindexed_scores_matrix, 1))]).T
        normalised_mat = self.__reindexed_scores_matrix - scores_mean
        A = normalised_mat.T
        # Using svd
        U, S, V = np.linalg.svd(A, full_matrices=full_matrix)
        # Reducing the dimensions
        self.__sliced = V.T[:, :dim_size]
        # special matrix multiplication to get the magnitudes.
        # element wise multiplication and summation row1 * col1, row2 * col2...so on
        self.__magnitude = np.sqrt(np.einsum('ij, ij -> i', self.__sliced, self.__sliced))
        print("SVD similarity calculated!")

    def get_svd_similar_users(self, user_handle, num_similar_users=TOP_CUTOFF_USERS):
        if not self.isValidUser(user_handle):
            print("Error - User not found!!")
            return None
        if self.__sliced is not None:
            index = np.where(self.index_values == user_handle)[0][0]  # we need index as int not array
            user_row = self.__sliced[index, :]
            similarity = np.dot(user_row, self.__sliced.T) / (self.__magnitude[index] * self.__magnitude)
            scores = pd.DataFrame(similarity, index=self.index_values)
            similar_users = scores.sort_values(by=[0], ascending=False)[1:num_similar_users]
            # Normalize between 0 to 1
            # similar_users[0] = (similar_users[0] - min(similar_users[0])) / (max(similar_users[0]) - min(similar_users[0]))
            return similar_users
        else:
            print("Error - SVD Similarity not calculated!!!!")

    # Function that taken in user handle as input and outputs most similar users based on pearson.
    def train_KNN_BaseLine(self):
        reader = Reader(rating_scale=(self.__dataframe[self.__values_column].min(),
                                      self.__dataframe[self.__values_column].max()))
        data = Dataset.load_from_df(self.__dataframe[[self.__index_column, self.__columns_column,
                                                      self.__values_column]], reader)
        sim_options = {'name': 'pearson_baseline'}
        self.__knn_algo = KNNBaseline(sim_options)
        # Train the algorithm on the trainset, and predict ratings for the testset
        self.__knn_algo.fit(data.build_full_trainset())

    def get_KNN_similar_users(self, user_handle, num_similar_users=TOP_CUTOFF_USERS):
        if not self.isValidUser(user_handle):
            print("Error - User not found!!")
            return None
        if self.__knn_algo is not None:
            index = np.where(self.index_values == user_handle)[0][0]  # we need index as int not array
            index_neighbor_users = self.__knn_algo.get_neighbors(index, k=num_similar_users)

            scores = pd.DataFrame(self.__knn_algo.sim[index].T, index=self.index_values)
            similar_users = scores.sort_values(by=[0], ascending=False)[1:num_similar_users]
            # Normalize between 0 to 1
            # similar_users[0] = (similar_users[0] - min(similar_users[0])) / (max(similar_users[0]) - min(similar_users[0]))
            return similar_users
        else:
            print("Error - KNN Similarity not calculated!!!!")

    def isValidUser(self, user_handle):
        return ((self.index_values == user_handle).sum() > 0)

    def __init__(self, parameters):
        self.__dataframe = parameters['dataframe'].copy()
        self.__index_column = parameters['index_column']
        # The column in dataframe which will be used to created columns in matrix
        self.__columns_column = parameters['columns_column']
        if 'values_column' in parameters:
            self.__values_column = parameters['values_column']
        else:
            self.__values_column = None
        self.__initialize_matrix()

    class UserNotFound(Exception):
        """Raise this execption when user is not found"""
        pass


user_assessment_similarity_measure = SimilarityMeasure({
    'dataframe': user_assessment_scores,
    'index_column': 'user_handle', 'columns_column': 'assessment_tag',
    'values_column': 'user_assessment_score'
})
user_assessment_similarity_measure.train_KNN_BaseLine()
user_assessment_similarity_measure.calculate_cosine_similarity()
user_assessment_similarity_measure.calculate_svd_similarity(dim_size=OPTIMIZED_DIM_SIZE_USER_ASSESSMENT)
user_assessment_similarity_measure.calculate_pearson_similarity()

user_interest_similarity_measure = SimilarityMeasure({
    'dataframe': user_interests[['user_handle', 'interest_tag']].drop_duplicates(),
    'index_column': 'user_handle', 'columns_column': 'interest_tag'
})
user_interest_similarity_measure.calculate_cosine_similarity()
user_interest_similarity_measure.calculate_svd_similarity(dim_size=OPTIMIZED_DIM_SIZE_USER_INTEREST)
user_interest_similarity_measure.train_KNN_BaseLine()

user_courses_merge = pd.merge(left=user_course_views, right=course_tags, on=['course_id'])
grouped_users_course_tags = user_courses_merge[['user_handle', 'course_tags', 'view_time_seconds']] \
    .groupby(['user_handle', 'course_tags'])
meaned_group_user_courses_tag = grouped_users_course_tags.agg('mean').reset_index()
user_courseview_tag_similarity_measure = SimilarityMeasure({
    'dataframe': meaned_group_user_courses_tag[['user_handle', 'course_tags',
                                                'view_time_seconds']],
    'index_column': 'user_handle', 'columns_column': 'course_tags',
    'values_column': 'view_time_seconds'
})
user_courseview_tag_similarity_measure.calculate_svd_similarity(dim_size=OPTIMIZED_DIM_SIZE_USER_COURSE_TAG)
user_courseview_tag_similarity_measure.calculate_cosine_similarity()
user_courseview_tag_similarity_measure.train_KNN_BaseLine()

grouped_users_courses = user_course_views[['user_handle', 'course_id', 'view_time_seconds']] \
    .groupby(['user_handle', 'course_id'])
meaned_group_users_courses = grouped_users_courses.agg('mean').reset_index()
user_courseview_similarity_measure = SimilarityMeasure({
    'dataframe': meaned_group_users_courses[['user_handle', 'course_id', 'view_time_seconds']],
    'index_column': 'user_handle', 'columns_column': 'course_id',
    'values_column': 'view_time_seconds'
})
user_courseview_similarity_measure.calculate_cosine_similarity()
user_courseview_similarity_measure.calculate_svd_similarity(dim_size=OPTIMIZED_DIM_SIZE_USER_COURSE)
user_courseview_similarity_measure.train_KNN_BaseLine()

grouped_users_course_level = user_course_views[['user_handle', 'level', 'view_time_seconds']] \
    .groupby(['user_handle', 'level'])
meaned_group_users_course_level = grouped_users_course_level.agg('mean').reset_index()
user_course_level_similarity_measure = SimilarityMeasure({
    'dataframe': meaned_group_users_course_level[['user_handle', 'level', 'view_time_seconds']],
    'index_column': 'user_handle', 'columns_column': 'level',
    'values_column': 'view_time_seconds'
})
user_course_level_similarity_measure.calculate_cosine_similarity()
user_course_level_similarity_measure.calculate_svd_similarity(dim_size=OPTIMIZED_DIM_SIZE_USER_COURSE_LEVEL)
user_course_level_similarity_measure.train_KNN_BaseLine()


class SimiliarUsers:
    """A warpper for the model that we will pickle and use for prediction"""

    def get_merged_scores(self, cosine_similarity, svd_similarity, pearson_similarity, knn_similarity, score_type,
                          weight):
        svd_similarity = pd.DataFrame(svd_similarity).reset_index()
        svd_similarity.columns = ['index', 'svd_' + score_type]
        cosine_similarity = pd.DataFrame(cosine_similarity).reset_index()
        cosine_similarity.columns = ['index', 'cosine_' + score_type]
        merged = pd.merge(left=svd_similarity, right=cosine_similarity, on='index')
        merged['total_' + score_type] = merged['svd_' + score_type] + merged['cosine_' + score_type]
        n_scores = 2
        if pearson_similarity is not None:
            pearson_similarity = pd.DataFrame(pearson_similarity).reset_index()
            pearson_similarity.columns = ['index', 'pearson_' + score_type]
            merged = pd.merge(left=merged, right=pearson_similarity, on='index')
            merged['total_' + score_type] = merged['total_' + score_type] + merged['pearson_' + score_type]
            n_scores = n_scores + 1
        merged['weighted_AVG_' + score_type] = (merged['total_' + score_type] * weight) / n_scores
        return merged

    def get_assessment_similarity(self, user_handle):
        if user_assessment_similarity_measure.isValidUser(user_handle):
            cosine_similarity = self.user_assessment_similarity_measure.get_cosine_similar_users(user_handle,
                                                                                                 self.N_USERS_TO_COMPARE)
            svd_similarity = self.user_assessment_similarity_measure.get_svd_similar_users(user_handle,
                                                                                           self.N_USERS_TO_COMPARE)
            pearson_similarity = self.user_assessment_similarity_measure.get_pearson_similar_users(user_handle,
                                                                                                   self.N_USERS_TO_COMPARE)
            knn_similarity = self.user_assessment_similarity_measure.get_KNN_similar_users(user_handle,
                                                                                           self.N_USERS_TO_COMPARE)
            return self.get_merged_scores(cosine_similarity, svd_similarity,
                                          pearson_similarity, knn_similarity, 'assessment', self.score_weights['A'])
        else:
            return None

    def get_interest_similarity(self, user_handle):
        if user_interest_similarity_measure.isValidUser(user_handle):
            cosine_similarity = self.user_interest_similarity_measure.get_cosine_similar_users(user_handle,
                                                                                               self.N_USERS_TO_COMPARE)
            svd_similarity = self.user_interest_similarity_measure.get_svd_similar_users(user_handle,
                                                                                         self.N_USERS_TO_COMPARE)
            pearson_similarity = self.user_interest_similarity_measure.get_pearson_similar_users(user_handle,
                                                                                                 self.N_USERS_TO_COMPARE)
            knn_similarity = self.user_interest_similarity_measure.get_KNN_similar_users(user_handle,
                                                                                         self.N_USERS_TO_COMPARE)
            return self.get_merged_scores(cosine_similarity, svd_similarity,
                                          pearson_similarity, knn_similarity, 'interest', self.score_weights['I'])
        else:
            return None

    def get_course_tag_similarity(self, user_handle):
        if user_courseview_tag_similarity_measure.isValidUser(user_handle):
            cosine_similarity = self.user_courseview_tag_similarity_measure.get_cosine_similar_users(user_handle,
                                                                                                     self.N_USERS_TO_COMPARE)
            svd_similarity = self.user_courseview_tag_similarity_measure.get_svd_similar_users(user_handle,
                                                                                               self.N_USERS_TO_COMPARE)
            pearson_similarity = self.user_courseview_tag_similarity_measure.get_pearson_similar_users(user_handle,
                                                                                                       self.N_USERS_TO_COMPARE)
            knn_similarity = self.user_courseview_tag_similarity_measure.get_KNN_similar_users(user_handle,
                                                                                               self.N_USERS_TO_COMPARE)
            return self.get_merged_scores(cosine_similarity, svd_similarity,
                                          pearson_similarity, knn_similarity, 'course_tag', self.score_weights['CVT'])
        else:
            return None

    def get_course_view_similarity(self, user_handle):
        if user_courseview_similarity_measure.isValidUser(user_handle):
            cosine_similarity = self.user_courseview_similarity_measure.get_cosine_similar_users(user_handle,
                                                                                                 self.N_USERS_TO_COMPARE)
            svd_similarity = self.user_courseview_similarity_measure.get_svd_similar_users(user_handle,
                                                                                           self.N_USERS_TO_COMPARE)
            pearson_similarity = self.user_courseview_similarity_measure.get_pearson_similar_users(user_handle,
                                                                                                   self.N_USERS_TO_COMPARE)
            knn_similarity = self.user_courseview_similarity_measure.get_KNN_similar_users(user_handle,
                                                                                           self.N_USERS_TO_COMPARE)
            return self.get_merged_scores(cosine_similarity, svd_similarity,
                                          pearson_similarity, knn_similarity, 'course_view', self.score_weights['CV'])
        else:
            return None

    def get_course_level_similarity(self, user_handle):
        if user_course_level_similarity_measure.isValidUser(user_handle):
            cosine_similarity = self.user_course_level_similarity_measure.get_cosine_similar_users(user_handle,
                                                                                                   self.N_USERS_TO_COMPARE)
            svd_similarity = self.user_course_level_similarity_measure.get_svd_similar_users(user_handle,
                                                                                             self.N_USERS_TO_COMPARE)
            pearson_similarity = self.user_course_level_similarity_measure.get_pearson_similar_users(user_handle,
                                                                                                     self.N_USERS_TO_COMPARE)
            knn_similarity = self.user_course_level_similarity_measure.get_KNN_similar_users(user_handle,
                                                                                             self.N_USERS_TO_COMPARE)
            return self.get_merged_scores(cosine_similarity, svd_similarity,
                                          pearson_similarity, knn_similarity, 'course_level', self.score_weights['CL'])
        else:
            return None

    def calculate_total_score(self, merged_similarity):
        total_columns = [s for s in merged_similarity.columns if 'AVG' in s]
        print(self.score_weights)
        merged_similarity['AVG_OF_ALL'] = merged_similarity[total_columns].sum(axis=1) / sum(
            self.score_weights.values())
        merged_similarity = merged_similarity.sort_values(by=['AVG_OF_ALL'], ascending=False)
        return merged_similarity

    def set_score_weights(self, new_weights):
        if new_weights is not None:
            self.score_weights = new_weights
        else:
            print("Invalid weights passed!!!!")

    def get_similar_users(self, user_handle):
        # store old weight because the weights dictionary will change if any score not present.
        temp_weights = self.score_weights.copy()
        merged_similarity = None

        user_assessment_similarity = self.get_assessment_similarity(user_handle)
        user_interest_similarity = self.get_interest_similarity(user_handle)

        if (user_assessment_similarity is not None) and (user_interest_similarity is not None):
            merged_similarity = pd.merge(user_assessment_similarity, user_interest_similarity, on='index')
        elif user_assessment_similarity is not None:
            del self.score_weights['I']
            merged_similarity = user_assessment_similarity
        elif user_interest_similarity is not None:
            del self.score_weights['A']
            merged_similarity = user_interest_similarity

        user_coursetag_similarity = self.get_course_tag_similarity(user_handle)
        if (user_coursetag_similarity is not None):
            merged_similarity = pd.merge(merged_similarity, user_coursetag_similarity, on='index')
        else:
            del self.score_weights['CVT']

        user_courseview_similarity = self.get_course_view_similarity(user_handle)
        if (user_courseview_similarity is not None):
            merged_similarity = pd.merge(merged_similarity, user_courseview_similarity, on='index')
        else:
            del self.score_weights['CV']

        user_courselevel_similarity = self.get_course_level_similarity(user_handle)
        if (user_courselevel_similarity is not None):
            merged_similarity = pd.merge(merged_similarity, user_courselevel_similarity, on='index')
        else:
            del self.score_weights['CL']

        if merged_similarity is not None:
            merged_similarity = self.calculate_total_score(merged_similarity)
            merged_similarity = merged_similarity.rename(index=str, columns={"index": "user_handle"})

        # set the weight back to original
        self.score_weights = temp_weights
        return merged_similarity

    def __init__(self, parameters):
        self.user_assessment_similarity_measure = parameters['user_assessment_similarity_measure']
        self.user_interest_similarity_measure = parameters['user_interest_similarity_measure']
        self.user_courseview_tag_similarity_measure = parameters['user_courseview_tag_similarity_measure']
        self.user_courseview_similarity_measure = parameters['user_courseview_similarity_measure']
        self.user_course_level_similarity_measure = parameters['user_course_level_similarity_measure']
        if 'score_weights' in parameters:
            self.score_weights = parameters['score_weights']
        else:
            # By default give equal weight to all the scores
            # A = Assessment, I = User Interest, CVT = Course view tags, CV = Course View, CL = Course Level
            self.score_weights = {'A': 1, 'I': 1, 'CVT': 1, 'CV': 1, 'CL': 1}
        self.N_USERS_TO_COMPARE = 10000


similarity_measures = {
    'user_assessment_similarity_measure': user_assessment_similarity_measure,
    'user_interest_similarity_measure': user_interest_similarity_measure,
    'user_courseview_tag_similarity_measure': user_courseview_tag_similarity_measure,
    'user_courseview_similarity_measure': user_courseview_similarity_measure,
    'user_course_level_similarity_measure': user_course_level_similarity_measure,
    'score_weights': {'A': 1, 'I': 1, 'CVT': 1, 'CV': 1, 'CL': 1}
}
similar_users_model = SimiliarUsers(similarity_measures)
similar_users_model.set_score_weights({'A': 1, 'I': 1, 'CVT': 2, 'CV': 1, 'CL': 1})
similar_users_df = similar_users_model.get_similar_users('1')
print(similar_users_df.head(5))
print("Again.............................")

pickle.dump(similar_users_model, open('model.pkl', 'wb'))
