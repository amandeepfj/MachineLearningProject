-- Created by Aman at 2/22/2019
-- "We are drowning in information, while starving for wisdom - E. O. Wilson"

-- This file loads csvs to sqlite.
-- database name = sqlite3 pluralsight.db
.mode csv
.import "C:/US Drive/MSBA/Pluralsight/pluralsight_ml_exercise/data_files_ml_engineer/user_assessment_scores.csv" user_assessment_scores
.import "C:/US Drive/MSBA/Pluralsight/pluralsight_ml_exercise/data_files_ml_engineer/user_course_views.csv" user_course_views
.import "C:/US Drive/MSBA/Pluralsight/pluralsight_ml_exercise/data_files_ml_engineer/course_tags.csv" course_tags
.import "C:/US Drive/MSBA/Pluralsight/pluralsight_ml_exercise/data_files_ml_engineer/user_interests.csv" user_interests

-- Check the counts to be sure that csv files are loaded correctly.
.schema user_assessment_scores
SELECT COUNT(*) FROM user_assessment_scores;

.schema user_course_views
SELECT COUNT(*) FROM user_course_views;

.schema course_tags
SELECT COUNT(*) FROM course_tags;

.schema user_interests
SELECT COUNT(*) FROM user_interests;