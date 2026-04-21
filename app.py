import streamlit as st
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

st.set_page_config(page_title='Fuel Efficiency Predictor', page_icon='🚗', layout='centered')

@st.cache_resource
def load_model():
    return joblib.load('fuel_efficiency_model.pkl')

model = load_model()

num_cols = ['Engine_Size_L','Horsepower','Weight_kg','Mileage_km_per_year','Fuel_Tank_Capacity','Age_of_Car','Service_Visits_Per_Year']
onehot_cols = ['Fuel_Type','Transmission','Car_Brand','Road_Type']
ordinal_cols = ['Owner_Type']

# Fit preprocessor from dataset used in training
@st.cache_resource
def load_preprocessor():
    df = pd.read_csv('final_mixed_dataset_12cols.csv')
    X = df.drop(columns=['Fuel_Efficiency_kmpl'])
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')),('scaler', StandardScaler())])
    cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('encoder', OneHotEncoder(handle_unknown='ignore'))])
    ord_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('encoder', OrdinalEncoder(categories=[['First','Second','Third']]))])
    pre = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, onehot_cols),
        ('ord', ord_pipeline, ordinal_cols)
    ])
    pre.fit(X)
    return pre

preprocessor = load_preprocessor()

st.title('🚗 Fuel Efficiency Predictor')
st.write('Predict fuel efficiency (km/l) based on vehicle details.')

with st.form('predict_form'):
    c1,c2 = st.columns(2)
    with c1:
        engine = st.number_input('Engine Size (L)', 0.8, 8.0, 2.0, 0.1)
        hp = st.number_input('Horsepower', 50, 1000, 120)
        weight = st.number_input('Weight (kg)', 600, 5000, 1500)
        mileage = st.number_input('Mileage per Year (km)', 1000, 100000, 15000)
        tank = st.number_input('Fuel Tank Capacity (L)', 20, 120, 45)
        age = st.number_input('Age of Car (years)', 0, 30, 5)
    with c2:
        visits = st.number_input('Service Visits / Year', 0, 12, 2)
        fuel = st.selectbox('Fuel Type', ['Petrol','Diesel','Electric','Hybrid','CNG'])
        trans = st.selectbox('Transmission', ['Manual','Automatic'])
        brand = st.selectbox('Car Brand', ['Toyota','Honda','Ford','BMW','Audi','Hyundai','Kia','Tata','Mahindra','Maruti'])
        road = st.selectbox('Road Type', ['City','Highway','Mixed'])
        owner = st.selectbox('Owner Type', ['First','Second','Third'])
    submitted = st.form_submit_button('Predict')

if submitted:
    row = pd.DataFrame([{
        'Engine_Size_L':engine,'Horsepower':hp,'Weight_kg':weight,
        'Mileage_km_per_year':mileage,'Fuel_Tank_Capacity':tank,
        'Age_of_Car':age,'Service_Visits_Per_Year':visits,
        'Fuel_Type':fuel,'Transmission':trans,'Car_Brand':brand,
        'Road_Type':road,'Owner_Type':owner
    }])
    Xp = preprocessor.transform(row)
    pred = model.predict(Xp)[0]
    st.success(f'Estimated Fuel Efficiency: {pred:.2f} km/l')
