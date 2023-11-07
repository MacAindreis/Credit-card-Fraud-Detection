import streamlit as st
import numpy as np
import pickle

# Path to the file for saving user feedback
feedback_file_path = 'C:/Users/MAC-DREWS/Documents/Datasets/creditCard/feedback.text'
# path to model - Logistic Regression
lr_model_file_path = 'C:/Users/MAC-DREWS/Documents/Datasets/creditCard/trained_credit_card_model.sav'
# path to model - Random Forest
rf_model_file_path = 'C:/Users/MAC-DREWS/Documents/Datasets/creditCard/trained_credit_modelRF.sav'
# path to model - SVM
svm_model_file_path = 'C:/Users/MAC-DREWS/Documents/Datasets/creditCard/trained_credit_modelSVM.sav'
# path to model - KNN
knn_model_file_path = 'C:/Users/MAC-DREWS/Documents/Datasets/creditCard/trained_credit_modelKNN.sav'


def app():
    st.title('Credit Card Fraud Detection')
    st.write('''
        This is the Credit Card Fraud Detection app created in Streamlit using the
        [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset.
        ''')
    st.write(
        'This app predicts the probability of a credit card transaction being fraudulent or not.')
    # st.write('Please select a specific machine learning algorithm from the sidebar to get started!')

    st.sidebar.title('Select Machine Learning Algorithm')
    st.sidebar.write(
        'Please select a machine learning algorithm from the following options:')
    ml_algorithm = st.sidebar.selectbox(
        'ML Algorithm', ('Logistic Regression', 'Random Forest', 'SVM', 'KNN'))

    if ml_algorithm == 'Random Forest':
        #getting the input data of the user
        st.write('Selected Algorithm: Random Forest')
        time_call = st.number_input("Time in Seconds")
        v1 = st.number_input("V1")
        v2 = st.number_input("V2")
        v3 = st.number_input("V3")
        v4 = st.number_input("V4")
        v5 = st.number_input("V5")
        v6 = st.number_input("V6")
        v7 = st.number_input("V7")
        v8 = st.number_input("V8")
        v9 = st.number_input("V9")
        v10 = st.number_input("V10")
        v11 = st.number_input("V11")
        v12 = st.number_input("V12")
        v13 = st.number_input("V13")
        v14 = st.number_input("V14")
        v15 = st.number_input("V15")
        v16 = st.number_input("V16")
        v17 = st.number_input("V17")
        v18 = st.number_input("V18")
        v19 = st.number_input("V19")
        v20 = st.number_input("V20")
        v21 = st.number_input("V21")
        v22 = st.number_input("V22")
        v23 = st.number_input("V23")
        v24 = st.number_input("V24")
        v25 = st.number_input("V25")
        v26 = st.number_input("V26")
        v27 = st.number_input("V27")
        v28 = st.number_input("V28")
        amount = st.number_input("Transaction Amount")

        amount = st.text_input("Transaction Amount")
    elif(ml_algorithm == 'Logistic Regression'):
        st.write('Selected Algorithm: Logistic Regression')
        time_call = st.number_input("Time in Seconds")
        v1 = st.number_input("V1")
        v2 = st.number_input("V2")
        v3 = st.number_input("V3")
        v4 = st.number_input("V4")
        v5 = st.number_input("V5")
        v6 = st.number_input("V6")
        v7 = st.number_input("V7")
        v8 = st.number_input("V8")
        v9 = st.number_input("V9")
        v10 = st.number_input("V10")
        v11 = st.number_input("V11")
        v12 = st.number_input("V12")
        v13 = st.number_input("V13")
        v14 = st.number_input("V14")
        v15 = st.number_input("V15")
        v16 = st.number_input("V16")
        v17 = st.number_input("V17")
        v18 = st.number_input("V18")
        v19 = st.number_input("V19")
        v20 = st.number_input("V20")
        v21 = st.number_input("V21")
        v22 = st.number_input("V22")
        v23 = st.number_input("V23")
        v24 = st.number_input("V24")
        v25 = st.number_input("V25")
        v26 = st.number_input("V26")
        v27 = st.number_input("V27")
        v28 = st.number_input("V28")
        amount = st.number_input("Transaction Amount")

    elif(ml_algorithm == 'SVM'):
        st.write('Selected Algorithm: SVM')
        time_call = st.number_input("Time in Seconds")
        v1 = st.number_input("V1")
        v2 = st.number_input("V2")
        v3 = st.number_input("V3")
        v4 = st.number_input("V4")
        v5 = st.number_input("V5")
        v6 = st.number_input("V6")
        v7 = st.number_input("V7")
        v8 = st.number_input("V8")
        v9 = st.number_input("V9")
        v10 = st.number_input("V10")
        v11 = st.number_input("V11")
        v12 = st.number_input("V12")
        v13 = st.number_input("V13")
        v14 = st.number_input("V14")
        v15 = st.number_input("V15")
        v16 = st.number_input("V16")
        v17 = st.number_input("V17")
        v18 = st.number_input("V18")
        v19 = st.number_input("V19")
        v20 = st.number_input("V20")
        v21 = st.number_input("V21")
        v22 = st.number_input("V22")
        v23 = st.number_input("V23")
        v24 = st.number_input("V24")
        v25 = st.number_input("V25")
        v26 = st.number_input("V26")
        v27 = st.number_input("V27")
        v28 = st.number_input("V28")
        amount = st.number_input("Transaction Amount")

    elif(ml_algorithm == 'KNN'):
        st.write('Selected Algorithm: KNN')
        time_call = st.number_input("Time in Seconds")
        v1 = st.number_input("V1")
        v2 = st.number_input("V2")
        v3 = st.number_input("V3")
        v4 = st.number_input("V4")
        v5 = st.number_input("V5")
        v6 = st.number_input("V6")
        v7 = st.number_input("V7")
        v8 = st.number_input("V8")
        v9 = st.number_input("V9")
        v10 = st.number_input("V10")
        v11 = st.number_input("V11")
        v12 = st.number_input("V12")
        v13 = st.number_input("V13")
        v14 = st.number_input("V14")
        v15 = st.number_input("V15")
        v16 = st.number_input("V16")
        v17 = st.number_input("V17")
        v18 = st.number_input("V18")
        v19 = st.number_input("V19")
        v20 = st.number_input("V20")
        v21 = st.number_input("V21")
        v22 = st.number_input("V22")
        v23 = st.number_input("V23")
        v24 = st.number_input("V24")
        v25 = st.number_input("V25")
        v26 = st.number_input("V26")
        v27 = st.number_input("V27")
        v28 = st.number_input("V28")
        amount = st.number_input("Transaction Amount")

    def FraudPrediction(input_data):
        #changing the input data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)
        #reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        if(ml_algorithm == 'Random Forest'):
            loaded_model = pickle.load(open(rf_model_file_path, 'rb'))
            prediction = loaded_model.predict(input_data_reshaped)
        elif(ml_algorithm == 'Logistic Regression'):
            loaded_model = pickle.load(open(lr_model_file_path, 'rb'))
            prediction = loaded_model.predict(input_data_reshaped)
        elif(ml_algorithm == 'SVM'):
            loaded_model = pickle.load(open(svm_model_file_path, 'rb'))
            prediction = loaded_model.predict(input_data_reshaped)
        elif(ml_algorithm == 'KNN'):
            loaded_model = pickle.load(open(knn_model_file_path, 'rb'))
            prediction = loaded_model.predict(input_data_reshaped)
        print(prediction)
        if(prediction[0] == 0):
            return 'You are not Fraud'
        else:
            return 'Sorry, You are Fraud'

    #code for prediction
    fraud = ''

    #creating a button for prediction
    if st.button('Fraud Prediction Result'):
        fraud = FraudPrediction([float(time_call), float(v1), float(v2), float(v3), float(v4), float(v5), float(v6), float(v7), float(v8), float(v9), float(v10), float(v11), float(v12), float(v13), float(
            v14), float(v15), float(v16), float(v17), float(v18), float(v19), float(v20), float(v21), float(v22), float(v23), float(v24), float(v25), float(v26), float(v27), float(v28), float(amount)])

    st.success(fraud)

#function to take user feedback


def feedback():
    st.header("User Feedback")
    user_feedback = st.text_area("Enter your feedback here:")
    if st.button("Submit Feedback"):
        with open(feedback_file_path, "a") as feedback_file:
            feedback_file.write(user_feedback + "\n")
        st.success("Feedback submitted successfully!")


app()
feedback()
