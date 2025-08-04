import pandas as pd
import numpy as np

#from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import streamlit as st

# Load the dataset
df = pd.read_csv('Financial_inclusion_dataset.csv')

# Information generrales du dataset
basic_info = {
    "shape": df.shape,
    "columns": df.dtypes.to_dict(),
    "missing_values": df.isnull().sum(),
    "duplicated_rows": df.duplicated().sum(),
    "descriptive_stats": df.describe()
}
# Affichage
basic_info

# Create a profiling report
#profile = ProfileReport(df, title='Financial Inclusion Profiling Report')
#profile.to_file("financial_inclusion_report.html")

# Drop columns that won't be useful for prediction
df.drop(['uniqueid', 'country', 'year'], axis=1, inplace=True)

# Convert binary columns to numerical
df['bank_account'] = df['bank_account'].map({'Yes': 1, 'No': 0})
df['cellphone_access'] = df['cellphone_access'].map({'Yes': 1, 'No': 0})

# Encode categorical features
categorical_cols = ['location_type', 'gender_of_respondent', 'relationship_with_head', 
                   'marital_status', 'education_level', 'job_type']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save label encoders for later use
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
    
# Split data
X = df.drop('bank_account', axis=1)
y = df['bank_account']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Save model
with open('financial_inclusion_model.pkl', 'wb') as f:
    pickle.dump(model, f)


# Load model and encoders
with open('financial_inclusion_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

st.title('Financial Inclusion Prediction App')

# Input fields
st.header('User Information')
location_type = st.selectbox('Location Type', ['Rural', 'Urban'])
cellphone_access = st.selectbox('Cellphone Access', ['Yes', 'No'])
household_size = st.number_input('Household Size', min_value=1, max_value=20, value=3)
age = st.number_input('Age', min_value=18, max_value=100, value=30)
gender = st.selectbox('Gender', ['Male', 'Female'])
relationship = st.selectbox('Relationship with Head', ['Head of Household', 'Spouse', 'Child', 'Other relative', 'Parent', 'Other non-relatives'])
marital_status = st.selectbox('Marital Status', ['Married/Living together', 'Single/Never Married', 'Widowed', 'Divorced/Seperated'])
education = st.selectbox('Education Level', ['No formal education', 'Primary education', 'Secondary education', 'Vocational/Specialised training', 'Tertiary education'])
job_type = st.selectbox('Job Type', ['Farming and Fishing', 'Self employed', 'Informally employed', 'Formally employed Government', 'Remittance Dependent', 'Formally employed Private'])

# In your input data preparation:
input_data = pd.DataFrame({
    'location_type': [location_type],
    'cellphone_access': [1 if cellphone_access == 'Yes' else 0],
    'household_size': [household_size],
    'age_of_respondent': [age],
    'gender_of_respondent': [gender],
    'relationship_with_head': [relationship],
    'marital_status': [marital_status],
    'education_level': [education],
    'job_type': [job_type]
})

# Then encode and predict:
for col in label_encoders:
    input_data[col] = label_encoders[col].transform(input_data[col])

prediction = model.predict(input_data)

# Make prediction
if st.button('Predict'):
    # Drop country and year as they're constant
    prediction = model.predict(input_data.drop(['country', 'year'], axis=1))
    probability = model.predict_proba(input_data.drop(['country', 'year'], axis=1))
    
    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.success(f'This person is likely to have a bank account (Probability: {probability[0][1]:.2%})')
    else:
        st.error(f'This person is unlikely to have a bank account (Probability: {probability[0][0]:.2%})')
    
    # Feature importance
    st.subheader('Key Factors Influencing This Prediction')
    feature_importance = pd.DataFrame({
        'Feature': model.feature_names_in_,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(5)
    
    st.bar_chart(feature_importance.set_index('Feature'))
