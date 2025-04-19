import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import zipfile
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set page config MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="U.S. Housing Market Analysis", layout="wide")

# Load the dataset from zip file on GitHub
@st.cache_data
def load_data():
    # GitHub raw content URL for the zip file
    zip_url = "https://github.com/Shamsfathalla/datascience/raw/main/datasets.zip"
    
    try:
        # Download the zip file
        response = requests.get(zip_url)
        response.raise_for_status()
        
        # Extract the zip file in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            # Find our dataset file in the zip
            csv_file_name = None
            for file in zip_ref.namelist():
                if "feature_engineered_dataset_capped_scaled.csv" in file:
                    csv_file_name = file
                    break
            
            if csv_file_name:
                with zip_ref.open(csv_file_name) as csv_file:
                    df = pd.read_csv(csv_file)
                    st.sidebar.success("Successfully loaded dataset from GitHub zip")
                    return df
            else:
                st.error("CSV file not found in the zip archive")
                return None
                
    except Exception as e:
        st.error(f"Failed to load data from GitHub: {str(e)}")
        return None

# Load the data
df = load_data()

if df is None:
    st.error("Critical Error: Could not load dataset. Please check the data source.")
    st.stop()

# Mapping for readability
area_type_map = {0: 'Rural', 1: 'Suburban', 2: 'Urban'}
city_type_labels = {
    0: 'Town',
    1: 'Small City',
    2: 'Medium City',
    3: 'Large City',
    4: 'Metropolis'
}

# Apply readable labels
df['area_type_label'] = df['area_type'].map(area_type_map)
df['city_type_label'] = df['city_type'].map(city_type_labels)

# Title
st.title("U.S. Housing Market Analysis")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", 
                         ["Home", 
                          "Regional Price Differences", 
                          "Bedrooms/Bathrooms Impact", 
                          "House Size by City Type", 
                          "Urban/Suburban/Rural Prices",
                          "House Size Predictor"])

# Home section
if section == "Home":
    st.header("Welcome to the U.S. Housing Market Analysis Dashboard")
    st.write("""
    This interactive dashboard provides insights into various aspects of the U.S. housing market. 
    Use the navigation panel on the left to explore different sections:
    
    - **Regional Price Differences**: Compare property prices across U.S. regions
    - **Bedrooms/Bathrooms Impact**: See how these features affect home prices
    - **House Size by City Type**: Explore average house sizes across different city types
    - **Urban/Suburban/Rural Prices**: Compare prices across different area types
    - **House Size Predictor**: Predict house size based on property features
    """)
    
    st.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
             caption="U.S. Housing Market Analysis", use_column_width=True)

# Regional Price Differences section
elif section == "Regional Price Differences":
    st.header("1. How do property prices differ between the different U.S. regions?")
    
    # Calculate average prices by region
    region_columns = ['region_Midwest', 'region_Northeast', 'region_South', 'region_West']
    region_avg_prices = df[region_columns].mul(df['price'], axis=0).sum() / df[region_columns].sum()
    region_avg_prices.index = ['Midwest', 'Northeast', 'South', 'West']
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=region_avg_prices.index, y=region_avg_prices.values, 
                palette=['blue', 'green', 'red', 'purple'], ax=ax)
    ax.set_title("Average Property Price by Region", fontsize=16)
    ax.set_xlabel("Region", fontsize=14)
    ax.set_ylabel("Average Price", fontsize=14)
    
    # Add value labels with 4 decimal places (no rounding)
    for i, value in enumerate(region_avg_prices.values):
        ax.text(i, value, f'{value:.4f}', ha='center', va='bottom', fontsize=12)
    
    st.pyplot(fig)
    
    st.write("""
    ### Key Insights:
    - West Region:
        - Property Price : The West has the highest average property price, driven by factors such as strong demand, desirable locations (e.g., coastal areas, major cities like Los Angeles, San Francisco), and high economic activity.
        - Population and Density : Despite having a moderate population density, the West's high property prices suggest that other factors, such as lifestyle preferences and economic opportunities, play a significant role in driving up costs.
    - Northeast Region :
        - Property Price : The Northeast has one of the highest property prices, second only to the West. This aligns with its high population density and extensive urban development, which typically correlates with higher property values.
        - Population and Density : The Northeast's high density reflects its concentration of major cities (e.g., New York City, Boston) and robust economic activity, contributing to elevated property prices.
    - South Region :
        - Property Price : The South has property prices slightly lower than the Northeast but higher than the Midwest. This suggests that while the South may not have the same level of urbanization or economic activity as the Northeast, it still benefits from growing regional economies and increasing demand for housing.
        - Population and Density : The South has a lower population density and average property size, indicating that factors beyond density (e.g., regional economic growth, migration patterns) influence property prices.
    - Midwest Region :
        - Property Price : The Midwest has the lowest average property price, reflecting lower demand and different market dynamics compared to other regions.
        - Population and Density : With lower population density and moderate property sizes, the Midwest appears to be less expensive overall. This could be due to slower economic growth, fewer major metropolitan areas, and less competition for housing.

    ### Final Answer
    - The property prices differ significantly between U.S. regions, with the following ranking from highest to lowest:
        - West (Highest average property price)
        - Northeast
        - South
        - Midwest (Lowest average property price)
    - These differences are influenced by a combination of factors, including population density, urban development, economic activity, and regional demand for housing. The West and Northeast, with their high densities and economic hubs, command the highest property prices, while the Midwest, with its lower density and slower economic growth, has the lowest property prices. The South falls in between, benefiting from moderate economic growth and increasing demand.
    """)

# Bedrooms/Bathrooms Impact section
elif section == "Bedrooms/Bathrooms Impact":
    st.header("2. How does the number of bedrooms and bathrooms affect home prices?")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bedrooms vs Price
    sns.lineplot(x='bed', y='price', data=df, marker='o', ax=ax1)
    ax1.set_title('Number of Bedrooms vs Price', fontsize=14)
    ax1.set_xlabel('Number of Bedrooms', fontsize=12)
    ax1.set_ylabel('Price', fontsize=12)
    
    # Add value labels for bedrooms
    bed_avg = df.groupby('bed')['price'].mean().reset_index()
    for i, row in bed_avg.iterrows():
        ax1.text(row['bed'], row['price'], f"${row['price']:,.0f}", ha='center', va='bottom', fontsize=10)
    
    # Bathrooms vs Price
    sns.lineplot(x='bath', y='price', data=df, marker='o', ax=ax2)
    ax2.set_title('Number of Bathrooms vs Price', fontsize=14)
    ax2.set_xlabel('Number of Bathrooms', fontsize=12)
    ax2.set_ylabel('Price', fontsize=12)
    
    # Add value labels for bathrooms
    bath_avg = df.groupby('bath')['price'].mean().reset_index()
    for i, row in bath_avg.iterrows():
        ax2.text(row['bath'], row['price'], f"${row['price']:,.0f}", ha='center', va='bottom', fontsize=10)
    
    st.pyplot(fig)
    
    st.write("""
    ### Key Insights:
    - Prices generally increase with more bedrooms and bathrooms, but the relationship isn't perfectly linear
    - The biggest price jumps occur when moving from:
        - 2 to 3 bedrooms
        - 1 to 2 bathrooms
    - Homes with 5+ bedrooms or 4+ bathrooms show more variability in pricing
    """)

# House Size by City Type section
elif section == "House Size by City Type":
    st.header("3. What is the average house size per city types in the U.S.?")
    
    # Calculate average house size by city type
    avg_house_size_city_type = df.groupby('city_type')['house_size'].mean().reset_index()
    avg_house_size_city_type['city_type_label'] = avg_house_size_city_type['city_type'].map(city_type_labels)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='city_type_label', y='house_size', data=avg_house_size_city_type, 
                palette='viridis', ax=ax, order=['Town', 'Small City', 'Medium City', 'Large City', 'Metropolis'])
    ax.set_title('Average House Size by City Type', fontsize=16)
    ax.set_xlabel('City Type', fontsize=14)
    ax.set_ylabel('Average House Size (sq ft)', fontsize=14)
    
    # Add value labels
    for i, row in avg_house_size_city_type.iterrows():
        ax.text(i, row['house_size'], f"{row['house_size']:,.0f} sq ft", ha='center', va='bottom', fontsize=12)
    
    st.pyplot(fig)
    
    st.write("""
    ### Key Insights:
    - **Towns** have the largest average house sizes
    - House sizes generally decrease as city size increases
    - **Metropolises** have the smallest average house sizes
    - This pattern reflects the land availability and density differences
    """)

# Urban/Suburban/Rural Prices section
elif section == "Urban/Suburban/Rural Prices":
    st.header("4. How do prices fluctuate between urban, suburban and rural cities?")
    
    # Calculate average price by area type
    avg_price_area = df.groupby('area_type_label')['price'].mean().reset_index()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='area_type_label', y='price', data=avg_price_area, 
                palette='rocket', order=['Rural', 'Suburban', 'Urban'], ax=ax)
    ax.set_title('Average Property Price by Area Type', fontsize=16)
    ax.set_xlabel('Area Type', fontsize=14)
    ax.set_ylabel('Average Price', fontsize=14)
    
    # Add value labels
    for i, row in avg_price_area.iterrows():
        ax.text(i, row['price'], f"${row['price']:,.0f}", ha='center', va='bottom', fontsize=12)
    
    st.pyplot(fig)
    
    st.write("""
    ### Key Insights:
    - **Urban** areas have the highest average property prices
    - **Suburban** areas follow urban areas in pricing
    - **Rural** areas have the lowest average prices
    - The urban premium reflects higher demand and limited space in cities
    """)

# House Size Predictor section
elif section == "House Size Predictor":
    st.header("House Size Predictor")
    st.write("""
    Predict the size of a house based on its characteristics.
    Adjust the sliders to input property features and see the predicted size.
    """)
    
    # Prepare data for modeling
    features = ['bed', 'bath', 'price', 'property_size', 'acre_lot', 'city_type', 'area_type']
    X = df[features]
    y = df['house_size']
    
    # Train a simple model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Display model performance
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.sidebar.write(f"Model Performance:")
    st.sidebar.write(f"- R² Score: {r2:.3f}")
    st.sidebar.write(f"- Mean Squared Error: {mse:,.0f}")
    
    # User inputs
    col1, col2 = st.columns(2)
    
    with col1:
        beds = st.slider("Number of Bedrooms", min_value=1, max_value=8, value=3)
        baths = st.slider("Number of Bathrooms", min_value=1, max_value=6, value=2)
        price = st.slider("Price ($)", min_value=50000, max_value=2000000, value=300000, step=50000)
    
    with col2:
        property_size = st.slider("Property Size (sq ft)", min_value=500, max_value=10000, value=2000, step=100)
        acre_lot = st.slider("Lot Size (acres)", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
        city_type = st.selectbox("City Type", options=list(city_type_labels.values()))
        area_type = st.selectbox("Area Type", options=list(area_type_map.values()))
    
    # Convert city_type and area_type back to numerical values
    city_type_num = [k for k, v in city_type_labels.items() if v == city_type][0]
    area_type_num = [k for k, v in area_type_map.items() if v == area_type][0]
    
    # Make prediction
    input_data = [[beds, baths, price, property_size, acre_lot, city_type_num, area_type_num]]
    predicted_size = model.predict(input_data)[0]
    
    # Display prediction
    st.subheader("Prediction Result")
    st.metric(label="Predicted House Size", value=f"{predicted_size:,.0f} square feet")
    
    # Show feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis', ax=ax)
    ax.set_title('Feature Importance for House Size Prediction', fontsize=16)
    st.pyplot(fig)
