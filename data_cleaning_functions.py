#!/usr/bin/env python
# coding: utf-8

# In[72]:


import pandas as pd                      # For data manipulation and analysis
import numpy as np                       # For numerical computations
import matplotlib.pyplot as plt          # For creating plots and visualizations
import seaborn as sns                    # For enhanced data visualizations
import scipy.stats as stats              # For statistical analysis
import re                                # For regular expressions
import logging
import sys
import os


# In[73]:


# Set up logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


# In[74]:


def load_csv_data(file_path):
    # Input validation
    if not isinstance(file_path, str):
        raise ValueError("Input 'file_path' should be a string representing the file path.")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    try:
        logging.info(f"Starting data processing for {file_path}...")
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return data
    except Exception as e:
        raise ValueError(f"An error occurred while loading data from {file_path}: {e}")


# In[75]:


def load_excel_data(file_path):
    # Input validation
    if not isinstance(file_path, str):
        raise ValueError("Input 'file_path' should be a string representing the file path.")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")
    
    try:
        data = pd.read_excel(file_path)
        return data
    except Exception as e:
        raise ValueError(f"An error occurred while loading data from {file_path}: {e}")


# In[76]:


def data_analysis_report(raw_data):
    # Initialize an empty report dictionary to store findings and recommendations
    report = {}
    
    # Check column labels and recommend converting them to snake_case
    snake_case_columns = [col.strip().lower().replace(' ', '_').str.replace('[^\w]', '', regex=True) for col in raw_data.columns]
    if raw_data.columns.tolist() != snake_case_columns:
        report['Column Labels'] = {
            'Issue': 'Column labels are not in snake_case format',
#             'Recommendation': f'Rename columns to snake_case format: {", ".join(snake_case_columns)}'
        }
        
    # Check for duplicate records and recommend removing duplicates
    if raw_data.duplicated().any():
        report['Duplicate Records'] = {
            'Issue': 'Duplicate records exist in the data',
            'Recommendation': 'Remove duplicate records using drop_duplicates()'
        }
    return report


# In[77]:


def save_data(cleaned_data, file_path):
    # Input validation
    if not isinstance(cleaned_data, pd.DataFrame):
        raise ValueError("Input 'cleaned_data' is not a valid DataFrame.")
    
    if not isinstance(file_path, str):
        raise ValueError("Input 'file_path' should be a string representing the file path.")
    
    try:
        cleaned_data.to_csv(file_path, index=False)
        print(f"Data saved to {file_path} successfully.")
    except Exception as e:
        raise ValueError(f"An error occurred while saving data to {file_path}: {e}")


# In[100]:


# To ensure consistency and ease of use, standardize the column names of the dataframe. 
def clean_column_names(raw_data):
    # Input validation
    if not isinstance(raw_data, pd.DataFrame):
        raise ValueError("Input 'raw_data' is not a valid DataFrame.")
    
    # Logging
    logging.info("Cleaning column names...")

    # Create a defensive copy of the DataFrame
    cleaned_data = raw_data.copy()
    
    # Clean column names
    cleaned_data.columns = cleaned_data.columns.str.strip().str.replace(' ', '_').str.lower().str.replace('[^\w]', '', regex=True)
    
    # Logging
    logging.info("Column names cleaned.")
    
    return cleaned_data


# In[79]:


# Renames the column names of the given date frame with according to the specified column_names dictionary
# Example column_names={'st': 'state'}

def rename_column_names(raw_data, column_names):
    if not isinstance(raw_data, pd.DataFrame) or raw_data.empty:
        raise ValueError("Input 'raw_data' must be a non-empty DataFrame.")
    
    if not isinstance(column_names, dict):
        raise ValueError("Input 'column_names' must be a dictionary.")
    
    existing_columns = set(raw_data.columns)
    
    new_columns = set(column_names.values())
    
    if not new_columns.isdisjoint(existing_columns):
        raise ValueError("New column names should not overlap with existing column names.")
    
    for key in column_names:
        if key not in existing_columns:
            raise ValueError(f"Column '{key}' does not exist in the DataFrame.")
    
    raw_data = raw_data.rename(columns=column_names)
    return raw_data


# In[80]:


def fix_numeric_column_types(data):
    # Input validation
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Input 'data' should be a non-empty DataFrame.")
    
    # Identify numeric columns
    numeric_columns = data.select_dtypes(include=[int, float]).columns
    
    # Convert numeric columns to appropriate numeric types
    for col in numeric_columns:
        try:
            data[col] = pd.to_numeric(data[col])
        except ValueError:
            print(f"Warning: Unable to convert '{col}' to numeric. It contains non-numeric values.")
    
    return data


# In[81]:


def identify_and_convert_numeric_column(data, column):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input 'data' must be a pandas DataFrame.")
    
    if column not in data.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    
    # Extract the column and remove any leading/trailing whitespaces
    column_data = data[column].str.strip()
    
    # Check if all values in the column are numeric
    if column_data.str.isnumeric().all():
        data[column] = pd.to_numeric(column_data)
    
    return data


# In[82]:


def identify_and_convert_all_numeric_columns(data):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input 'data' must be a pandas DataFrame.")
    
    for column in data.columns:
        data = identify_and_convert_numeric_column(data, column)
    
    return data


# In[ ]:





# In[101]:


#Remove rows with any missing values (NaN) from a pandas DataFrame.

def remove_empty_raws(data):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        # Logging
        logging.info("Removing rows with missing values (NaN)...")

        # Remove rows with all missing values (NaN)
        data.dropna(how='all', inplace=True)

        # Logging
        logging.info("Rows with missing values (NaN) removed.")

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[84]:


# Drop rows with null values in specific columns.

def drop_raws_with_na_values(data, columns):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        # Check if 'columns' is a list of valid column names in 'data'
        invalid_columns = set(columns) - set(data.columns)
        if invalid_columns:
            raise ValueError(f"Invalid columns: {invalid_columns}. The DataFrame does not have these columns.")

        # Drop rows with NA values in the specified columns
        data.dropna(subset=columns, inplace=True)

        # Return the cleaned DataFrame
        return data

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[85]:


def get_duplicate_rows(data, columns=None):
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        if columns is not None and not isinstance(columns, list):
            raise ValueError("Input 'columns' must be a list of column names.")

        if columns is None:
            duplicate_rows = data[data.duplicated()]
        else:
            if not set(columns).issubset(data.columns):
                missing_columns = set(columns) - set(data.columns)
                raise ValueError(f"Columns {missing_columns} do not exist in the DataFrame.")

            duplicate_rows = data[data.duplicated(subset=columns)]

        return duplicate_rows
    
    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[106]:


# Drop duplicate rows based on specific columns.

def drop_duplicates(data, columns, keep):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        # Check if 'columns' is a list of valid column names in 'data'
        invalid_columns = set(columns) - set(data.columns)
        if invalid_columns:
            raise ValueError(f"Invalid columns: {invalid_columns}. The DataFrame does not have these columns.")

        # Check if 'keep' argument is valid
        valid_keep_values = ['first', 'last', False]
        if keep not in valid_keep_values:
            raise ValueError(f"Invalid value for 'keep': {keep}. Valid values are 'first', 'last', and False.")

        # Logging
        logging.info("Dropping duplicate rows based on specific columns...")

        # Drop duplicate rows based on the specified columns and 'keep' option
        data.drop_duplicates(subset=columns, keep=keep, inplace=True)

        # Logging
        logging.info("Duplicate rows dropped based on specific columns.")

        # Return the cleaned DataFrame
        return data

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[107]:


# Replace inconsistent values with their correct counterparts
def replace_inconsistent_values(data, column, mapping):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        # Check if 'column' is a valid column name in 'data'
        if column not in data.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

        # Check if 'mapping' is a dictionary
        if not isinstance(mapping, dict):
            raise ValueError("'mapping' argument must be a dictionary.")

        # Replace inconsistent values in the specified column with their correct counterparts
        data[column] = data[column].replace(mapping)

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[88]:


# Fill null values in a specific column of a pandas DataFrame with either the mean or median.

def fill_null_with_mean_or_median(data, column, method='mean'):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        # Check if 'column' is a valid column in the DataFrame
        if column not in data.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

        # Check if 'method' is valid
        if method not in ['mean', 'median']:
            raise ValueError("'method' must be one of 'mean' or 'median'.")

        # Get the data type of the column
        dtype = data[column].dtype

        # Check if the column is numeric (int64 or float64) for mean or median calculation
        if dtype != 'int64' and dtype != 'float64':
            raise ValueError(f"Column '{column}' is not numeric (int64 or float64).")

        # Fill null values based on the specified method
        if method == 'mean':
            data[column].fillna(data[column].mean(), inplace=True)
        elif method == 'median':
            data[column].fillna(data[column].median(), inplace=True)

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[89]:


# Fill null values in a pandas DataFrame with appropriate values based on column data types.


def fill_all_null_values(data):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        # Loop through each column
        for col in data.columns:
            # Get the data type of the column
            dtype = data[col].dtype

            # Fill null values based on data type
            if dtype == 'object':
                data[col].fillna(data[col].mode()[0], inplace=True)  # Fill with mode
            elif dtype == 'int64' or dtype == 'float64':
                data[col].fillna(data[col].mean(), inplace=True)     # Fill with mean

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[90]:


def fill_null_with_previous_or_next_value(data, column, method='previous'):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        # Check if 'column' is a valid column in the DataFrame
        if column not in data.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

        # Check if 'method' is valid
        if method not in ['previous', 'next']:
            raise ValueError("'method' must be one of 'previous' or 'next'.")

        # Fill null values based on the specified method
        if method == 'previous':
            data[column].fillna(method='ffill', inplace=True)
        elif method == 'next':
            data[column].fillna(method='bfill', inplace=True)

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[91]:


def fill_nulls_in_dataset_with_previous_or_next(data, method='previous'):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        # Check if 'method' is valid
        if method not in ['previous', 'next']:
            raise ValueError("'method' must be one of 'previous' or 'next'.")

        # Iterate through each column and fill null values with the specified method
        for column in data.columns:
            fill_null_with_previous_or_next_value(data, column, method=method)

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[92]:


# Check if there are null values in a pandas DataFrame and return the list of columns with null values.

def check_null_values(data):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        # Get the list of columns with null values
        columns_with_nulls = data.columns[data.isnull().any()].tolist()

        return columns_with_nulls

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[93]:


# Get all rows from a pandas DataFrame that have null values in the specified list of columns.
# If the 'columns' list is empty, it will consider all columns for checking null values.

def get_rows_with_null_values(data, columns=[]):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        # Check if 'columns' is a list
        if not isinstance(columns, list):
            raise ValueError("Input 'columns' must be a list of column names.")

        # If 'columns' is empty, consider all columns for checking null values
        if not columns:
            rows_with_nulls = data[data.isnull().any(axis=1)]
        else:
            rows_with_nulls = data[data[columns].isnull().any(axis=1)]

        return rows_with_nulls

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[ ]:





# In[111]:


def clean_data(raw_data):
    logging.info(f"Starting cleaning data...")
    
    raw_data = clean_column_names(raw_data)
    remove_empty_raws(raw_data)
    #raw_data = drop_duplicates(raw_data)
    logging.info(f"Data cleaning completed.")
    
    return raw_data


# In[95]:


# Function to check if a string is in snake_case format
def is_snake_case(string):
    return re.match(r'^[a-z_][a-z0-9_]*$', string)


# In[96]:


# Define a function to detect outliers using the z-score method
def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs((data - data.mean()) / data.std())
    return z_scores > threshold


# In[97]:


# Identify the majority data type for a column
def identify_data_type(column_values):
    # Create a Series from the column values
    series = pd.Series(column_values)

    # Get the data type counts using value_counts()
    data_type_counts = series.apply(type).value_counts()

    # Get the majority data type
    majority_data_type = data_type_counts.idxmax().__name__
    
    return majority_data_type


# In[98]:


# Identify the list of columns with inconsistent data types
def identify_inconsistent_data_types(df):
    remove_empty_raws(df)
    inconsistent_columns = {}

    for col in df.columns:
        series = df[col]

        # Get the data type counts using value_counts()
        data_type_counts = series.apply(type).value_counts()
        
        # Get the majority data type
        majority_data_type = data_type_counts.idxmax()
        
        # Check if the majority data type is different from the actual column data type
        if majority_data_type != type(series.iloc[0]):
            inconsistent_columns[col] = majority_data_type

    return inconsistent_columns


# In[99]:


def data_analysis_report(raw_data):
    # Initialize an empty report dictionary to store findings and recommendations
    report = {}
    
    # Basic check: Is the dataset empty?
    if raw_data.empty:
        report['Empty Dataset'] = {
            'issue': 'The dataset is empty',
            'recommendation': 'Check the data source and load the dataset properly'
        }
        return report
    
    # Check column labels and recommend converting them to snake_case
    non_snake_case_columns = [col for col in raw_data.columns if not is_snake_case(col)]
    if non_snake_case_columns:
        report['Non-Snake Case Columns'] = {
            'columns': non_snake_case_columns,
            'count': len(non_snake_case_columns),
            'issue': 'Column labels are not in snake_case format',
            'recommendation': f'Rename columns to snake_case format: {", ".join(non_snake_case_columns)}'
        }
    
    # Check for duplicate rows
    duplicates = raw_data.duplicated()
    if duplicates.any():
        report['Duplicate Rows'] = {
            'count': duplicates.sum(),
            'issue': 'Duplicate rows found in the dataset',
            'recommendation': 'Remove or handle duplicate rows appropriately'
        }
    
    # Check for missing values in each column
    missing_values_check = raw_data.isnull().sum()
    if missing_values_check.any():
        columns_with_missing_values = missing_values_check[missing_values_check > 0].index.tolist()
        report['Missing Values'] = {
            'columns': columns_with_missing_values,
            'count': missing_values_check.to_dict(),
            'issue': 'Missing values found in the dataset',
            'recommendation': 'Remove or impute missing values'
        }
        
    
    # Check for outliers in numerical columns
    numerical_columns = raw_data.select_dtypes(include=[np.number]).columns
    outliers_columns = []
    for numerical_column in numerical_columns:
        outliers_check = detect_outliers_zscore(raw_data[numerical_column])
        
        if outliers_check.any():
            outliers_columns.append(numerical_column)
            
    if outliers_columns:
        report['Outliers Detected'] = {
            'columns': outliers_columns,
            'count': len(outliers_columns),
            'issue': 'Outliers detected in the dataset',
            'recommendation': 'Handle outliers appropriately (e.g., remove or transform)'
        }
            
    # Check data types of columns
    inconsistent_data_types = identify_inconsistent_data_types(raw_data)
    
    report['Incorrect Data Types'] = {
        'columns': inconsistent_data_types,
        'count': len(inconsistent_data_types),
        'issue': 'Inconsistent or incorrect data types',
        'recommendation': 'Ensure data types are appropriate and consistent'
    }
    
    return pd.DataFrame.from_dict(report, orient='index')


# In[ ]:





# In[ ]:




