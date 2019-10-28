# Company: Cantel Medical
# Organization: IT
# Department: Advanced Analytics
# Author: Garrett Eichhorn

"""

This python script is intended to load a pre-defined Machine Learning Model to make predictions. Specifically, I'm
employing a Supervised Classification model using Natural Language Processing (NLP) to make predictions for incoming
ServiceNow (SNOW) tickets. The model learns from text fields using Logistic Regression, mapping important features
to defined categories: Portfolio and Assignment Group.

This script will make a prediction, evaluate the significance of produced prediction, and decide to ignore or PUT the
prediction back into the SNOW platform.

"""

# Load the Model as LogRegModel
import joblib
LogRegModel = joblib.load('C:\\Users\garrett.eichhorn\PycharmProjects\MachineLearning\Prediction Pipeline\LogisticRegression_model.pkl')

########################################################################################################################

import requests
import pandas as pd
import json
import numpy as np

np.set_printoptions(formatter={'float_kind':'{:f}'.format})
pd.set_option('display.max_colwidth', -1)

########################################################################################################################

user = 'garrett.eichhorn.api'
pwd = 'Garrett1234'

headers = {"Content-Type":"application/json",
           "Accept":"application/json",
           'accept-encoding': 'gzip, deflate, br'}
today_url = 'https://cantel.service-now.com/api/now/table/incident?' \
            'sysparm_query=opened_atONToday%40javascript%3Ags.beginningOfToday()' \
            '%40javascript%3Ags.endOfToday()&sysparm_fields=sys_id'

# Function to retrieve new incident records from ServiceNow
def retrieve_incident_records(url):

    response = requests.get(url, auth=(user, pwd), headers=headers)
    api_data = response.json()
    query = pd.DataFrame(api_data['result'])

    return query

# Function to combine text columns from a given dataframe
def combine_text_columns(dataframe):

    # Drop non-text columns that are in the df
    text_data = dataframe

    # Replace nans with blanks
    text_data.fillna("", inplace=True)

    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)

# Function to create dummy labels from parsed data file (contains text only)
def create_dummies(file, column):

    # Read the excel file (downloaded initially from MongoDB) which contains 27,000 records
    full_file = pd.read_excel(file, index_col='number')

    # Keep relevant columns and convert type(), drop records without feature: portfolio
    text_columns = full_file[[column, 'description', 'short_description']]
    text_columns[column] = text_columns[column].astype('category')
    cat_df = text_columns.dropna()

    # Create dummy labels for model processing, can be added to return label
    dummy_labels = pd.get_dummies(cat_df[[column]], prefix_sep='_')

    # Keep a "full" dataframe for quick analysis of both labels and text, can add to return label
    thicc_dataframe = pd.concat([cat_df, dummy_labels], axis=1)

    return dummy_labels

dummy_labels_assignmentGroup = create_dummies("C:\\Users\garrett.eichhorn\PycharmProjects\MachineLearning\Gather Input Data\Data File for ML Pipeline_AssignmentGroup.xlsx", "Assignment_group")
dummy_labels_portfolio = create_dummies("C:\\Users\garrett.eichhorn\PycharmProjects\MachineLearning\Gather Input Data\Data File for ML Pipeline_AssignmentGroup.xlsx", "u_portfolio")

# Function to evaluate model performance
def evaluate_model (inc_list, model, p):

    result_df = pd.DataFrame()
    count_non_sig = 0

    for i in inc_list.values:

        new_url = "https://cantel.service-now.com/api/now/table/incident/" + i[0] + "?sysparm_fields=short_description%2Cdescription"

        # Instantiate GET request
        response = requests.get(new_url, auth=(user, pwd), headers=headers)

        api_data = json.loads(response.text)
        query_df = pd.DataFrame(api_data)
        query_df_transposed = query_df.T

        # Combine text
        query_text_vector = combine_text_columns(query_df_transposed)

        # Run text through MODEL
        q_predict = model.predict(query_text_vector)
        prob = model.predict_proba(query_text_vector)

        # Store results through dataframe for easy insight
        predicted_result_probabilities = np.concatenate((q_predict, prob))
        prediction_df = pd.DataFrame(columns=dummy_labels_portfolio.columns, data=predicted_result_probabilities)
        prediction_df_transpose = prediction_df.T
        prediction_df_transpose.columns = ['Significance_bool', 'Probability_score']

        # I want to count how many incidents were not significant according to the model
        for j in q_predict:
            if not any(j):
                count_non_sig += 1

        # I only want the predicted result, dropping all other values
        pred_result = prediction_df_transpose[prediction_df_transpose > 0]
        pred_result = pred_result.dropna()

        # When the model identifies multiple classifiers, we want to grab the category with the highest probability
        if len(pred_result) > 1:
            empty_list = []
            for i in pred_result['Probability_score']:
                empty_list.append(i)

            higher_prob = np.amax(empty_list)
            pred_result = pred_result[pred_result['Probability_score'] == higher_prob]

        # When the model identifies a classifier with significance, we want the probability to be higher than a specified value
        if not pred_result.empty:
            if pred_result.iloc[0]['Probability_score'] >= p:

                pred_result['Text'] = str(query_text_vector)
                result_df = result_df.append(pred_result)

    count_sig = 0

    count_inc = len(inc_list)

    if count_inc > 0:

        sig_inc = count_inc - count_non_sig
        count_sig += sig_inc

    print("Percent of tickets found significant: " + str(count_sig / count_inc))

    return result_df

########################################################################################################################

today_SNOW_tickets = retrieve_incident_records(today_url)

e = evaluate_model(today_SNOW_tickets, LogRegModel, .5)

e.to_excel("C:\\Users\garrett.eichhorn\Desktop\\today's_tix.xlsx")
