import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import zipfile
import os
import requests

# Path to store the downloaded zip file and the extracted model
zip_file_path = 'model.zip'
extracted_model_path = 'price_model.pkl'

# Function to download and unzip the model
def download_and_unzip_model():
    if not os.path.exists(extracted_model_path):
        # Replace with your model's raw GitHub URL (the URL for raw file)
        raw_github_url = 'https://github.com/yourusername/yourrepo/raw/main/model.zip'

        # Download the zip file
        with requests.get(raw_github_url) as r:
            with open(zip_file_path, 'wb') as f:
                f.write(r.content)

        # Unzip the file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall()

# Unzip the model if it hasn't been extracted yet
download_and_unzip_model()

# Load dataset and model
df = pd.read_csv("feature_engineered_dataset.csv")
model = joblib.load(extracted_model_path)

st.title("US Real Estate Dashboard & Price Predictor")

st.markdown("##  Visualizations")

# Q1: Price vs Region
st.subheader("1. Average Price by Region")
region_cols = ['region_Midwest', 'region_Northeast', 'region_South', 'region_West']
region_map = {col: col.split('_')[1] for col in region_cols}
df["region"] = df[region_cols].idxmax(axis=1).map(region_map)
avg_region = df.groupby("region")["price"].mean().sort_values()
fig1, ax1 = plt.subplots()
sns.barplot(x=avg_region.index, y=avg_region.values, ax=ax1)
ax1.set_ylabel("Average Price")
st.pyplot(fig1)

# Q2: Bed/Bath vs Price
st.subheader("2. Effect of Bedrooms/Bathrooms on Price")
fig2, ax2 = plt.subplots()
sns.lineplot(data=df, x="bed_bath_ratio", y="price", ax=ax2)
st.pyplot(fig2)

# Q3: Property Size by City Type
st.subheader("3. Property Size by City Type")
fig3, ax3 = plt.subplots()
sns.boxplot(data=df, x="city_type", y="property_size", ax=ax3)
ax3.set_xlabel("City Type")
ax3.set_ylabel("Property Size")
st.pyplot(fig3)

# Q4: Price by Area Type
st.subheader("4. Price by Area Type")
fig4, ax4 = plt.subplots()
sns.lineplot(data=df, x="area_type", y="price", ax=ax4)
ax4.set_xlabel("Area Type")
st.pyplot(fig4)

# 🔮 Prediction section
st.markdown("## Predict Property Price")
with st.form("predict_form"):
    bed = st.slider("Bedrooms", 1, 10, 3)
    bath = st.slider("Bathrooms", 1, 10, 2)
    acre_lot = st.number_input("Acre Lot", min_value=0.0, max_value=10.0, value=0.5)
    house_size = st.number_input("House Size (scaled)", min_value=0.0, max_value=1.0, value=0.2)
    city_type = st.selectbox("City Type", [0, 1, 2, 3, 4])
    area_type = st.selectbox("Area Type", [0, 1, 2])
    population_2024 = st.number_input("Population 2024", value=50000)
    density = st.number_input("Population Density", value=1500.0)
    region = st.selectbox("Region", ["Midwest", "Northeast", "South", "West"])
    
    submit = st.form_submit_button("Predict Price")

    if submit:
        bed_bath_ratio = bed / bath if bath != 0 else 0
        property_size = house_size + acre_lot
        region_vals = [1 if region == r else 0 for r in ["Midwest", "Northeast", "South", "West"]]
        price_per_bed = 0  # these engineered features can be zero or estimated
        price_per_bath = 0
        price_per_bed_bath = 0
        
        input_data = pd.DataFrame([[
            bed, bath, acre_lot, house_size, population_2024, density,
            city_type, area_type, property_size, bed_bath_ratio,
            *region_vals, price_per_bed, price_per_bath, price_per_bed_bath
        ]], columns=[
            "bed", "bath", "acre_lot", "house_size", "population_2024", "density",
            "city_type", "area_type", "property_size", "bed_bath_ratio",
            "region_Midwest", "region_Northeast", "region_South", "region_West",
            "price_per_bed", "price_per_bath", "price_per_bed_bath"
        ])

        prediction = model.predict(input_data)[0]
        st.success(f"🏷 Predicted Scaled Price: {prediction:.4f}")
