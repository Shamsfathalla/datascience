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

# Load the dataset from zip file on GitHub with caching
@st.cache_data
def load_data():
    zip_url = "https://github.com/Shamsfathalla/datascience/raw/main/datasets.zip"
    
    try:
        response = requests.get(zip_url)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            csv_file_name = next((f for f in zip_ref.namelist() if "feature_engineered_dataset_capped_scaled.csv" in f), None)
            
            if csv_file_name:
                with zip_ref.open(csv_file_name) as csv_file:
                    df = pd.read_csv(csv_file)
                    
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
                    
                    return df
            else:
                return None
                
    except Exception as e:
        return None

# Load the data and handle UI feedback
df = load_data()
if df is not None:
    st.toast("Successfully loaded dataset from GitHub zip", icon="✅")
else:
    st.error("Failed to load data from GitHub or CSV file not found in the zip archive")
    st.stop()

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
             caption="U.S. Housing Market Analysis", use_container_width=True)

# Regional Price Differences section
elif section == "Regional Price Differences":
    st.header("1. How do property prices differ between the different U.S. regions?")

    # Define region names and dummy columns
    region_names = ['Midwest', 'Northeast', 'South', 'West']
    region_columns = ['region_Midwest', 'region_Northeast', 'region_South', 'region_West']

    # Define color palette
    region_colors = ['blue', 'green', 'red', 'purple']

    # Calculate region-based averages
    region_avg_prices = df[region_columns].mul(df['price'], axis=0).sum() / df[region_columns].sum()
    region_avg_population_2024 = df[region_columns].mul(df['population_2024'], axis=0).sum() / df[region_columns].sum()
    region_avg_density = df[region_columns].mul(df['density'], axis=0).sum() / df[region_columns].sum()
    
    # Set proper indices
    region_avg_prices.index = region_names
    region_avg_population_2024.index = region_names
    region_avg_density.index = region_names

    # Set Seaborn style
    sns.set_style("whitegrid")

    # Create figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Bar plot for average prices
    sns.barplot(x=region_names, y=region_avg_prices.values, palette=region_colors, ax=axes[0])
    axes[0].set_title("Average Property Price by Region")
    axes[0].set_xlabel("Region")
    axes[0].set_ylabel("Average Price")
    for i, v in enumerate(region_avg_prices.values):
        axes[0].text(i, v, f'{v:.4f}', ha='center', va='bottom')
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.4f}'))

    # Bar plot for population
    sns.barplot(x=region_names, y=region_avg_population_2024.values, palette=region_colors, ax=axes[1])
    axes[1].set_title("Average Population in 2024 by Region")
    axes[1].set_xlabel("Region")
    axes[1].set_ylabel("Average Population")
    for i, v in enumerate(region_avg_population_2024.values):
        axes[1].text(i, v, f'{v:.4f}', ha='center', va='bottom')
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.4f}'))

    # Bar plot for density
    sns.barplot(x=region_names, y=region_avg_density.values, palette=region_colors, ax=axes[2])
    axes[2].set_title("Average Density by Region")
    axes[2].set_xlabel("Region")
    axes[2].set_ylabel("Average Density")
    for i, v in enumerate(region_avg_density.values):
        axes[2].text(i, v, f'{v:.4f}', ha='center', va='bottom')
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.4f}'))

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    st.pyplot(fig)
    
    st.write("""
    ### Key Insights:
    - West Region:
        - Property Price: The West has the highest average property price, driven by factors such as strong demand, desirable locations (e.g., coastal areas, major cities like Los Angeles, San Francisco), and high economic activity.
        - Population and Density: Despite having a moderate population density, the West's high property prices suggest that other factors, such as lifestyle preferences and economic opportunities, play a significant role in driving up costs.
    - Northeast Region:
        - Property Price: The Northeast has one of the highest property prices, second only to the West. This aligns with its high population density and extensive urban development, which typically correlates with higher property values.
        - Population and Density: The Northeast's high density reflects its concentration of major cities (e.g., New York City, Boston) and robust economic activity, contributing to elevated property prices.
    - South Region:
        - Property Price: The South has property prices slightly lower than the Northeast but higher than the Midwest. This suggests that while the South may not have the same level of urbanization or economic activity as the Northeast, it still benefits from growing regional economies and increasing demand for housing.
        - Population and Density: The South has a lower population density and average property size, indicating that factors beyond density (e.g., regional economic growth, migration patterns) influence property prices.
    - Midwest Region:
        - Property Price: The Midwest has the lowest average property price, reflecting lower demand and different market dynamics compared to other regions.
        - Population and Density: With lower population density and moderate property sizes, the Midwest appears to be less expensive overall. This could be due to slower economic growth, fewer major metropolitan areas, and less competition for housing.

    ### Final Answer
    - The property prices differ significantly between U.S. regions, with the following ranking from highest to lowest:
        - West (Highest average property price)
        - Northeast
        - South
        - Midwest (Lowest average property price)
    - These differences are influenced by a combination of factors, including population density, urban development, economic activity, and regional demand for housing. The West and Northeast, with their high density and economic hubs, command the highest property prices, while the Midwest, with its lower density and slower economic growth, has the lowest property prices. The South falls in between, benefiting from moderate economic growth and increasing demand.
    """)
    
# Bedrooms/Bathrooms Impact section
elif section == "Bedrooms/Bathrooms Impact":
    st.header("2. How does the number of bedrooms and bathrooms affect home prices?")

    # Validate required columns
    required_cols = ['bed', 'bath', 'bed_bath_ratio', 'price']
    if not all(col in df.columns for col in required_cols):
        st.error("Missing columns: " + ", ".join([col for col in required_cols if col not in df.columns]))
        st.stop()
    
    # Function to add text labels at line intersections
    def add_labels_to_lineplot(ax, data, x_col, y_col):
        # Get unique x values
        x_values = sorted(data[x_col].unique())  # Sort for consistent plotting
        for x in x_values:
            y = data[data[x_col] == x][y_col].mean()  # Calculate mean price for x
            ax.text(x, y, f'{y:.4f}', ha='center', va='bottom', fontsize=12)
    
    # Set Seaborn style for better aesthetics
    sns.set_style("whitegrid")
    
    # Create a figure with three subplots in vertical arrangement
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))  # 3 rows, 1 column
    
    # Plot 1: Bed vs Price - Added marker='o' for circles
    sns.lineplot(x='bed', y='price', data=df, ax=axes[0], marker='o', markersize=8)
    add_labels_to_lineplot(axes[0], df, 'bed', 'price')
    axes[0].set_title('Bedrooms vs Price')
    axes[0].set_xlabel('Number of Bedrooms')
    axes[0].set_ylabel('Price')
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.4f}'))
    
    # Plot 2: Bath vs Price - Added marker='s' for squares
    sns.lineplot(x='bath', y='price', data=df, ax=axes[1], marker='s', markersize=8)
    add_labels_to_lineplot(axes[1], df, 'bath', 'price')
    axes[1].set_title('Bathrooms vs Price')
    axes[1].set_xlabel('Number of Bathrooms')
    axes[1].set_ylabel('Price')
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.4f}'))
    
    # Plot 3: Bed_Bath_Ratio vs Price - Added marker='D' for diamonds
    sns.lineplot(x='bed_bath_ratio', y='price', data=df, ax=axes[2], marker='D', markersize=8)
    add_labels_to_lineplot(axes[2], df, 'bed_bath_ratio', 'price')
    axes[2].set_title('Bed/Bath Ratio vs Price')
    axes[2].set_xlabel('Bed to Bath Ratio')
    axes[2].set_ylabel('Price')
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.4f}'))
    axes[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    st.write("""
    ### Key Insights:
    - Number of Bedrooms: Increasing the number of bedrooms generally increases the home price, but the marginal increase in price diminishes as the number of bedrooms grows. Larger homes become less expensive per bedroom.
    - Number of Bathrooms: Similarly, increasing the number of bathrooms raises the home price, with diminishing returns as the number of bathrooms increases. Larger homes become less expensive per bathroom.
    - Bed/Bath Ratio: A balanced ratio of bedrooms to bathrooms (around 0.55) tends to maximize home prices. Homes with an imbalanced ratio (too many bedrooms relative to bathrooms or vice versa) may have lower prices.
        - Location and Area Type:
            - Smaller cities and rural areas tend to have fewer bedrooms and bathrooms compared to larger cities and urban areas.
            - However, the optimal bed/bath ratio (around 0.55) remains a key driver of home prices across all locations.
            - Urban areas, despite having more bedrooms and bathrooms, may have slightly lower ratios due to space constraints, but the principle of balance still applies.
    ### Final Answer:
    - The number of bedrooms and bathrooms positively affects home prices, but the impact diminishes as these numbers increase. A balanced bed/bath ratio (around 0.55) is optimal for maximizing home prices. However, it is important to note that bedrooms and bathrooms are not the only factors influencing home prices. These variables are continuous and likely interact with other features (e.g., location, property size, and amenities), which should be considered in a comprehensive pricing model.
    """)

# House Size by City Type section
elif section == "House Size by City Type":
    st.header("3. What is the average house size per city types in the U.S.?")
    
    # Calculate average house size by city type
    avg_house_size_city_type = df.groupby('city_type_label')['house_size'].mean().reset_index()
    
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
    
    # Train a simple model (with caching)
    @st.cache_resource
    def train_model():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model, X_test, y_test
    
    model, X_test, y_test = train_model()
    
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
