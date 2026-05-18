import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys

#page configuration
st.set_page_config(
    page_title = 'Credit Risk Predictor',
    page_icon  = '🏦',
    layout     = 'wide',
)

@st.cache_resource
def load_models():
    """Load preprocessor and model once, cache for the session."""
    preprocessor = joblib.load('outputs/models/preprocessor.pkl')
    model        = joblib.load('outputs/models/best_model.pkl')
    return preprocessor, model

preprocessor, model = load_models()

#header
st.title('Credit Risk Prediction System')
st.markdown(
    'Enter a loan applicant\'s details below to predict whether '
    'they are a **Good** or **Bad** credit risk.'
)
st.divider()

#input form
st.subheader('Applicant Details')

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('**Personal Information**')
    age = st.slider('Age', min_value=18, max_value=75, value=30, step=1)
    sex = st.selectbox('Sex', options=['male', 'female'])
    housing = st.selectbox('Housing', options=['own', 'free', 'rent'])

with col2:
    st.markdown('**Loan Details**')
    credit_amount = st.number_input(
        'Credit Amount (DM)',
        min_value=250, max_value=20000, value=3000, step=100
    )
    duration = st.slider(
        'Loan Duration (months)',
        min_value=4, max_value=72, value=24, step=1
    )
    purpose = st.selectbox('Purpose', options=[
        'car', 'furniture/equipment', 'radio/TV',
        'domestic appliances', 'repairs', 'education',
        'business', 'vacation/others'
    ])

with col3:
    st.markdown('**Financial Background**')
    saving_accounts = st.selectbox('Saving Accounts', options=[
        'little', 'moderate', 'quite rich', 'rich', 'NA'
    ])
    checking_account = st.selectbox('Checking Account', options=[
        'little', 'moderate', 'rich', 'NA'
    ])
st.divider()

#Prediction
predict_btn = st.button('Predict Credit Risk', type='primary', use_container_width=True)

if predict_btn:
    # 1. Assembles inputs into a DataFrame (same structure as training data)
    input_data = pd.DataFrame([{
        'Age':              age,
        'Sex':              sex,
        'Housing':          housing,
        'Saving accounts':  saving_accounts if saving_accounts != 'NA' else np.nan,
        'Checking account': checking_account if checking_account != 'NA' else np.nan,
        'Credit amount':    credit_amount,
        'Duration':         duration,
        'Purpose':          purpose,
    }])

    # 2. Run through the preprocessor
    input_processed = preprocessor.transform(input_data)

    # 3. Get prediction and probability
    prediction = model.predict(input_processed)[0]
    probability = model.predict_proba(input_processed)[0]

    bad_prob  = round(probability[1] * 100, 1)
    good_prob = round(probability[0] * 100, 1)

    # 4. Display result
    st.subheader('Prediction Result')

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if prediction == 0:
            st.success('GOOD CREDIT RISK')
            st.markdown(f'The model predicts this applicant is likely to **repay** the loan.')
        else:
            st.error('BAD CREDIT RISK')
            st.markdown(f'The model predicts this applicant is likely to **default** on the loan.')

    with res_col2:
        st.metric(label='Good Credit Probability', value=f'{good_prob}%')
        st.metric(label='Bad Credit Probability',  value=f'{bad_prob}%')
        st.progress(int(good_prob))

    # 5. Confidence note
    st.divider()
    if bad_prob < 30:
        st.info('High confidence, model is fairly certain about this prediction.')
    elif bad_prob < 50:
        st.warning('Borderline case, additional manual review is recommended.')
    else:
        st.error('High default risk, this application carries significant risk.')

