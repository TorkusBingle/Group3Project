#%%writefile appteam3.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
from datetime import datetime
import plotly.express as px

migrant_App = pd.read_csv('migrant_clean.csv', index_col=None)
timeseriesdf = pd.read_csv('timeseriesdf.csv')
#setting the index as a date for the timeseries
timeseriesdf['date'] = pd.to_datetime(timeseriesdf['date'])
timeseriesdf.set_index('date', inplace=True)

column_names = migrant_App.columns
print(column_names)

st.set_page_config(page_title="Migrant Web App", page_icon=":tada:", layout="wide")

# Header Section
st.subheader("Hi everyone, we are Team 3")
st.title("Migrant Data Analysis")
st.write("This is data analysis on Migrant Dataset from Team 3")

# Sidebar
st.sidebar.header("User Inputs")
planned_migration_date = st.sidebar.date_input("Input planned migration date", value="today", min_value=None, max_value=None, format="YYYY-MM-DD")

migration_route = st.sidebar.selectbox("Select Migration Route", timeseriesdf["migration route"].unique())

# region_of_origin = st.sidebar.selectbox("Select Region of Origin", migrant_App["region of origin"].unique())
# number_of_males = st.sidebar.number_input("Number of Males", min_value=0)
# incident_year = st.sidebar.slider("Select Incident Year", 2014, 2023)

# Main content
st.header("Migrant Data Prediction")

# Group data by "Region of Origin"
# grouped_data = migrant_App.groupby("region of origin group").agg({
#     "number of males": "sum",
#     "number of females": "sum",
#     "number of children": "sum",
#     "total number of dead and missing": "sum"
# })

# # Display the aggregated data
# st.write("Aggregated Data by Region:")
# st.write(grouped_data)

# # Plot the results (you can customize the chart type)
# st.bar_chart(grouped_data)


# User inputs
st.write("User Inputs:")
st.write(f"Planned Migration Date: {planned_migration_date}")
# st.write(f"Incident Year: {incident_year}")
# st.write(f"Region of Origin: {region_of_origin}")
# st.write(f"Number of Males: {number_of_males}")

# # Filter data based on user inputs
# filtered_data = migrant_App[(migrant_App["incident year"] == incident_year) &
#                             (migrant_App["region of origin group"] == region_of_origin) &
#                             (migrant_App["number of males"] == number_of_males)]


# if not filtered_data.empty:
#     # Display data
#     st.subheader("Analysis Results")
#     st.write("number of dead:", filtered_data["number of dead"].values[0])
#     st.write("estimated number of missings:", filtered_data["minimum estimated number of missing"].values[0])
#     st.write("Number of Dead and Missing:", filtered_data["total number of dead and missing"].values[0])
#     st.write("Number of Survivors:", filtered_data["number of survivors"].values[0])
#     st.write("Number of Females:", filtered_data["number of females"].values[0])
#     st.write("Number of Children:", filtered_data["number of children"].values[0])
#     st.write("Cause of Death:", filtered_data["cause of death category"].values[0])
#     st.write("Country of Death:", filtered_data["extracted country"].values[0])
# else:
#     st.warning("No data available for the selected inputs.")

# Additional features..............

#timeseries model 
st.subheader("Time Series Model")
#function that returns the level of the migration route inputted
def getLevelOfRoute(route, timeseriesdf):
    level = timeseriesdf[timeseriesdf['migration route'] == route]['label_level'].values[0]
    return level

#run the function with the migration route inputted   
ts_level = getLevelOfRoute(migration_route, timeseriesdf)

st.write(f"The migration route was classified as a danger {ts_level}.")

#function to get all the df entries with the same level 
def getClusterLabel(level, timeseries):
    return (timeseries[timeseries['label_level'] == level])

#function that extracts all the routes in the same cluster and groups them by the target variable.
def preprocess_level_timeseries(level, timeseries):
    # Get the 'level' timeseries
    level_timeseries = getClusterLabel(level, timeseries)
    
    # Drop the 'date' column
    level_timeseries = level_timeseries.drop(['date.1'], axis=1)
    
    # Group by date and sum the 'total number of dead and missing'
    level_timeseries = level_timeseries.groupby(level_timeseries.index)['total number of dead and missing'].sum()
    
    return level_timeseries

leveldf = preprocess_level_timeseries(ts_level, timeseriesdf)


#the parameters will depend on the level selected 
#if ts_level = 'level1' then app_order = (1, 1, 1) 
#if ts_level = 'level2' or ts_level = 'level5' then app_order = (1, 0, 0) 
#if ts_level = 'level3' or ts_level = 'level4' then app_order = (0, 1, 1) 
if ts_level == 'level1':
    app_order = (1, 1, 1)
elif ts_level == 'level2' or ts_level == 'level5':
    app_order = (1, 0, 0)
elif ts_level == 'level3' or ts_level == 'level4':
    app_order = (0, 1, 1)

#function that gets the number of periods for the sarima model 

def calculate_months_difference(date1, date2):
    date1 = pd.to_datetime(date1, format="%Y-%m-%d")
    date2 = pd.to_datetime(date2, format="%Y-%m-%d")

    rdelta = relativedelta(date2, date1)
    months_difference = rdelta.years * 12 + rdelta.months

    return months_difference

last_date = leveldf.index[-1]
target_date = planned_migration_date
months_difference = calculate_months_difference(last_date, target_date)


#function that sets the sarima timeseries model 
def sarima_forecast(level_timeseries,forecast_months, order, seasonal_order, plot_title="SARIMA Forecast"):
    # Create the SARIMA model
    level_sarima_model = sm.tsa.SARIMAX(level_timeseries, order=order, seasonal_order=seasonal_order)
    level_sarima_model_fit = level_sarima_model.fit()

    # Make forecasts
    forecasts = level_sarima_model_fit.get_forecast(steps=forecast_months)
    predicted_values = forecasts.predicted_mean
    predicted_values.index = pd.date_range(start=last_date, periods=forecast_months, freq='M')




    return predicted_values
#run the sarima function
predicted_values = sarima_forecast(leveldf, months_difference, order=app_order, seasonal_order=(1, 1, 1, 12), plot_title="Migrant Incident Forecast")


#function to retrieve the output of the timeseries for the inputted date
def get_values_for_year_month(indexes, values, year, month):
    matching_values = []
    for i, date_index in enumerate(indexes):
        if date_index.year == year and date_index.month == month:
            matching_values.append(values[i])
    return matching_values

# Example usage:
year_to_find = planned_migration_date.year
month_to_find = planned_migration_date.month
matching_values = get_values_for_year_month(predicted_values.index, predicted_values, year_to_find, month_to_find)
st.write(f"Estimated number of incidents for the planned migration date: {matching_values}")



# Plot the observed and forecasted values
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(leveldf.index, leveldf, label='Historical Data')
ax.plot(predicted_values.index, predicted_values, label='SARIMA Forecast', color='red')
ax.legend()
ax.set_title("Migrant Incident Forecast")
st.pyplot(fig)

# Create a time slider
time_range = st.slider("Select a time range", 0, 14, (0, 23))
 
# Update the datetime slider based on the selected time range
start_date = datetime(2014, 1, 1, time_range[0])
end_date = datetime(2025, 12, 30)
#end_date = start_date + timedelta(hours=time_range[1] - time_range[0])
 
# selected_date = slider_placeholder.slider(
#     "Select a date range",
#     min_value=start_date,
#     max_value=end_date,
#     value=(start_date, end_date),
#     step=timedelta(hours=1),
# )

#link to slider code i found https://docs.kanaries.net/topics/Python/streamlit-datetime-slider



#visualizing COD
st.title('Cause of Death per migration route')

# Creates route selection dropdown
migration_routes = migrant_App['migration route'].unique()
selected_route = st.selectbox('Please Select a Migration Route', migration_routes)

# Filters the data based on the selected migration route
filtered_data = migrant_App[migrant_App['migration route'] == selected_route]

# Creates the histogram
st.subheader(f'Causes of Death for {selected_route}')
fig = px.histogram(filtered_data, x='cause of death category')
st.plotly_chart(fig)



# Footer
st.sidebar.text("© 2023 Migrant Data Analysis App")

!streamlit run appteam3.py --server.port=8080 --browser.serverAddress='0.0.0.0'
