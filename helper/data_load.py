import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_data(config):
    data_set = pd.read_csv(config.data_path)
    return data_set

def file_missing_data(dataset, config):
    for i in dataset.columns:
        if dataset[i].dtype == 'object':
            if config.fill_obj_mode:
                dataset[i].fillna(dataset[i].mode()[0], inplace=True)
            elif config.fill_num_mean:
                dataset[i].fillna(dataset[i].mean(), inplace=True)
            else:
                return NotImplementedError("This method is not supported yet")

def encode_cat_data(dataset, config):
    """
    This function will encode the categorical data in the dataset
    """
    label_encoders = {}
    categorical_columns = dataset.select_dtypes(include=['object']).columns.drop(config.index_column)
    for i in categorical_columns:
        le = LabelEncoder()
        dataset[i] = le.fit_transform(dataset[i])
        label_encoders[i] = le
    return label_encoders

def normalize_data(dataset, config):
    scaler = MinMaxScaler()
    numeric_columns = dataset.select_dtypes(include=[np.number]).columns.drop(config.target_column)
    dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])
    return scaler

# use the newly defined functions to load the data
if __name__ == "__main__":
    config = Config()
    data_set = load_data(config)
    file_missing_data(data_set, config)
    label_encoders = encode_cat_data(data_set, config)
    scaler = normalize_data(data_set, config)

    # Calculate mean and standard deviation of continuous features
    continuous_mean = data_set[numeric_columns].mean().values
    continuous_std = data_set[numeric_columns].std().values


    # Splitting the dataset into training and validation sets
    X_categ_train, X_categ_test, X_cont_train, X_cont_test, y_train, y_test = train_test_split(
        X_categ, X_cont, y, test_size=0.2, random_state=42)