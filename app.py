import streamlit as st
import pandas as pd
import folium
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_folium import st_folium 
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Housing Price Dubai UAE.csv")
    return df

df = load_data()

# Streamlit Sidebar
st.sidebar.header("Filters")

# Price Range Filter
min_price, max_price = int(df["price"].min()), int(df["price"].max())
price_range = st.sidebar.slider("Select Price Range", min_value=min_price, max_value=max_price, value=(min_price, max_price))
df = df[(df["price"] >= price_range[0]) & (df["price"] <= price_range[1])]

# Neighborhood Filter
neighborhoods = df["neighborhood"].unique()
selected_neighborhood = st.sidebar.selectbox("Select Neighborhood", ["All"] + list(neighborhoods))
if selected_neighborhood != "All":
    df = df[df["neighborhood"] == selected_neighborhood]


# Folium Map with Prices
st.write("### Property Price Map")
m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=11)

for _, row in df.iterrows():
    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=f"Price: AED {row['price']:,}\nSize: {row['size_in_sqft']} sqft",
        icon=folium.Icon(color="blue"),
    ).add_to(m)

st_folium(m, width=800, height=500)

# Train XGBoost Model for Price Prediction
st.write("### Train an XGBoost Model to Predict Housing Prices")

# Select relevant features
features = ["size_in_sqft", "no_of_bedrooms", "no_of_bathrooms"]
target = "price"

# Encode categorical variables
df["neighborhood_encoded"] = LabelEncoder().fit_transform(df["neighborhood"])
features.append("neighborhood_encoded")

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

# Show model performance
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Prediction on User Input
st.write("### Predict Housing Price for a Custom Input")
size = st.number_input("Enter Size in Sqft", min_value=300, max_value=10_000, value=1200)
bedrooms = st.number_input("Enter Number of Bedrooms", min_value=0, max_value=5, value=2)
bathrooms = st.number_input("Enter Number of Bathrooms", min_value=1, max_value=5, value=2)
selected_area = st.selectbox("Select Neighborhood", list(df["neighborhood"].unique()))

# Convert neighborhood to encoded value
neighborhood_code = LabelEncoder().fit_transform(df["neighborhood"])
selected_area_code = neighborhood_code[list(df["neighborhood"].unique()).index(selected_area)]

# Predict price
input_data = np.array([[size, bedrooms, bathrooms, selected_area_code]])
predicted_price = model.predict(input_data)[0]

st.write(f"**Predicted Price:** AED {predicted_price:,.2f}")

# Add prediction marker to map
folium.Marker(
    location=[df[df["neighborhood"] == selected_area]["latitude"].mean(),
              df[df["neighborhood"] == selected_area]["longitude"].mean()],
    popup=f"Predicted Price: AED {predicted_price:,.2f}\nSize: {size} sqft",
    icon=folium.Icon(color="red"),
).add_to(m)

st_folium(m, width=800, height=500)
