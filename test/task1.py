import copy

from static.config import *
from static.process import *
from analysis.descriptive_analysis import *
from analysis.exploratory_analysis import *
from analysis.linear_model import *
from analysis.tree_model import *
from analysis.kernel_model import *
from analysis.bayes_model import *
from analysis.neural_model import *

'''
    Data Preprocessing
'''

data_preprocessing_info = {}

# Obtain the raw data
df = load_data()

df, new_info = preprocess_raw_data_filtering(copy.deepcopy(df))
data_preprocessing_info.update(new_info)

# Save the info of data preprocessing to a csv format file
pd.DataFrame([data_preprocessing_info]).to_csv("./data/data_preprocessing_info.csv", index=False)

'''
    Descriptive Analysis
'''

descriptive_analysis_info = {}

# Create images of the distribution of the number of each variable
variable_distribution(copy.deepcopy(df))

# Data transformation
df, new_info = data_transformation(copy.deepcopy(df))
descriptive_analysis_info.update(new_info)

# Get descriptive indicators and filtered data
df, new_info = get_descriptive_indicators_related(copy.deepcopy(df))
descriptive_analysis_info.update(new_info)

# Save the info of descriptive analysis to a csv format file
pd.DataFrame([descriptive_analysis_info]).to_csv("./data/descriptive_analysis_info.csv", index=False)

'''
    Exploratory analysis
'''

# Get the standardized data
array = get_standardized_data(copy.deepcopy(df))

exploratory_analysis_info = {}

# Principal component analysis
projected_array, new_info = pca(copy.deepcopy(df))
exploratory_analysis_info.update(new_info)

# K-means
# new_info = k_means(preprocessing.scale(copy.deepcopy(projected_array)))
# exploratory_analysis_info.update(new_info)

# Save the info of exploratory analysis to a csv format file
pd.DataFrame([exploratory_analysis_info]).to_csv("./data/exploratory_analysis_info.csv", index=False)

'''
    Predictive Analysis
'''

predictive_analysis_info = {}

# segment the whole dataset into total train dataset and test dataset
x_train_and_validate, x_test, y_train_and_validate, y_test = split_dataset(array)

# segment train dataset into train datasets and validate datasets
train_and_validate_data_list = k_fold_cross_validation_data_segmentation(x_train_and_validate, y_train_and_validate)

'''
    Linear Model
'''

# Linear regression
# new_info = linear_regression(x_train_and_validate, y_train_and_validate, x_test, y_test, train_and_validate_data_list, "grid_search", "Lasso")
# predictive_analysis_info.update(new_info)

# Polynomial regression
new_info = polynomial_regression(x_train_and_validate, y_train_and_validate, x_test, y_test, train_and_validate_data_list, "grid_search")
predictive_analysis_info.update(new_info)

# Logistic regression
new_info = logistic_regression(x_train_and_validate, y_train_and_validate, x_test, y_test, train_and_validate_data_list, "grid_search")
predictive_analysis_info.update(new_info)

'''
    Tree Model
'''

# Decision tree
new_info = decision_tree_classifier(x_train_and_validate, y_train_and_validate, x_test, y_test)
predictive_analysis_info.update(new_info)

# Random Forest classifier
new_info = random_forest_classifier(x_train_and_validate, y_train_and_validate, x_test, y_test)
predictive_analysis_info.update(new_info)

# xgboost classifier
new_info = xgboost_classifier(x_train_and_validate, y_train_and_validate, x_test, y_test)
predictive_analysis_info.update(new_info)

'''
    Kernel Model
'''

# svm classification
new_info = svm_classification(x_train_and_validate, y_train_and_validate, x_test, y_test)
predictive_analysis_info.update(new_info)

'''
    Bayes Model
'''

# Naive bayes classification
new_info = naive_bayes_classification(x_train_and_validate, y_train_and_validate, x_test, y_test)
predictive_analysis_info.update(new_info)

'''
    Neural Model
'''

ann(copy.deepcopy(df))

'''
    Distance Model
'''



















