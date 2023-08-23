from sys import displayhook
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import folium
from folium.plugins import HeatMap


# Reading CSV file
path_to_data_csv = Path.cwd().parent / "data" / "_properties_data.csv"
data_prop = pd.read_csv(path_to_data_csv)


# changes possible slots without values to NaN values
data_prop = data_prop.replace(r'^s*$', float('NaN'), regex=True)

# delete  rows where 'price' value are NaN
data_prop.dropna(subset=['price'], inplace=True)

# delete rows where 'region' value is NaN
data_prop.dropna(subset=['region'], inplace=True)

data_prop[['transactionType', 'transactionSubtype', 'type', 'subtype', 'locality', 'street', 'condition', 'kitchen']] = data_prop[['transactionType', 'transactionSubtype', 'type',
                                                                                                                                   'subtype', 'locality', 'street', 'condition', 'kitchen']].applymap(lambda x: str(x).capitalize())  # change string values which were written in upper case into capitilized.
data_prop['type'] = data_prop['type'].replace(
    r'_group$', '', regex=True)  # remove all "_group" in the "type" column

data_prop = data_prop.drop_duplicates(subset=['latitude', 'longitude', 'type', 'subtype', 'price', 'district',
                                      'locality', 'street', 'number', 'box', 'floor', 'netHabitableSurface'], keep='first')  # Cleaning from duplicates

data_prop['kitchen'] = data_prop.kitchen.replace(['Usa_installed', 'Installed', 'Semi_equipped',
                                                 'Hyper_equipped', 'Usa_hyper_equipped', 'Usa_semi_equipped'], 'Installed')  # Clearing kitchen column
data_prop['kitchen'] = data_prop.kitchen.replace(
    ['Usa_uninstalled', 'Not_installed'], 'Not_installed')
# Changing all 'NaN' values to 'No_information' values
data_prop = data_prop.fillna(value='No_information')
# Changing all str 'Nan', 'NaN' values to 'No_information'
data_prop = data_prop.replace(['Nan', 'NaN'], 'No_information')

# Changing type of postal code data into integer
data_prop['postalCode'] = data_prop['postalCode'].astype('int64')

# Cleaning Energy Class data
data_prop['epcScore'] = data_prop.epcScore.replace(
    ['G_A++', 'C_B', 'G_F', 'G_A', 'E_A', 'D_C'], 'No_information')


# Save clean data to CSV file
path_to_save_csv = Path.cwd().parent / "data" / "_properties_data_clean.csv"
data_prop.to_csv(path_to_save_csv)


# establishing Color Brewer palette
custom_palette = sns.color_palette("Paired", 12)


# Generating histogram of properties prices in Belgium
sns.displot(data_prop['price'], color=custom_palette[5])
plt.xlim(0, 1.25e6)
plt.xlabel('Price, euros')
plt.ylabel('Ammount of properties')
plt.title('Histogram of properties prices in Belgium')
path_to_save_1 = Path.cwd().parent / "output" / "plot_1.png"
plt.savefig(path_to_save_1)
plt.show()


# Generating histogram of apartment prices in Belgium
apartment_prices = data_prop[data_prop['type'] ==
                             'Apartment']['price']  # Filter apartment prices
sns.displot(apartment_prices, color=custom_palette[3])
plt.xlim(0, 1.25e6)
plt.xlabel('Price, euros')
plt.ylabel('Ammount of apartmnets')
plt.title('Histogram of apartment prices in Belgium')
path_to_save_2 = Path.cwd().parent / "output" / "plot_2.png"
plt.savefig(path_to_save_2)
plt.show()


# Generating histogram of house prices in Belgium
house_prices = data_prop[data_prop['type'] ==
                         'House']['price']  # Filter house prices
sns.displot(house_prices, color=custom_palette[1])
plt.xlim(0, 1.25e6)
plt.xlabel('Price, euros')
plt.ylabel('Ammount of houses')
plt.title('Histogram of house prices in Belgium')
path_to_save_3 = Path.cwd().parent / "output" / "plot_3.png"
plt.savefig(path_to_save_3)
plt.show()


# Generating a barplot for 2 types of appartment for sale in all provinces

prop_junction = data_prop.value_counts(['province', 'type']).reset_index()

sns.barplot(data=prop_junction,
            x=prop_junction['province'], y=prop_junction['count'], hue='type')
plt.xlabel('Provinces')
plt.ylabel('Ammount of apartments and houses for sale in Belgium')
plt.title('Bar plot of properties for sale in Belgian provices')
plt.xticks(rotation=90)
path_to_save_4 = Path.cwd().parent / "output" / "plot_4.png"
plt.savefig(path_to_save_4)
plt.show()


# Generating a Bar Plot of avarage prices of properties sotrted on provinces
# Calculate the average price for each province
average_prices = data_prop.groupby('province')['price'].mean()

# Sort the provinces based on average price in descending order
sorted_provinces = average_prices.sort_values(ascending=False).index

# Plot the bar plot
sns.barplot(x='province', y='price', data=data_prop, order=sorted_provinces,
            estimator=lambda x: x.mean(), palette=custom_palette)
plt.xlabel('Provinces')
plt.ylabel('Average prices of the properties (euros)')
plt.title('Bar plot of average prices of properties sorted by provinces')
plt.xticks(rotation=90)
plt.tight_layout()  # Adjust the layout to prevent labels from being cut off
path_to_save_5 = Path.cwd().parent / "output" / "plot_5.png"
plt.savefig(path_to_save_5)
plt.show()


# Generating a Bar Plot of avarage prices of properties splited on Apartments and Houses sotrted on provinces
prop_junction_new = data_prop.groupby(['province', 'type'])[
    'price'].mean().reset_index()
# Sort the provinces based on average price in descending order
sorted_provinces_new = prop_junction_new.groupby(
    'province')['price'].mean().sort_values(ascending=False).index
# Plot the bar plot
sns.barplot(x='province', y='price', hue='type', data=prop_junction_new,
            order=sorted_provinces_new, estimator=lambda x: x.mean())
plt.xlabel('Provinces')
plt.ylabel('Average prices of the properties (euros)')
plt.title('Bar plot of average prices of properties sorted by provinces')
plt.xticks(rotation=90)
plt.tight_layout()  # Adjust the layout to prevent labels from being cut off
plt.legend(title='Property Type')
path_to_save_6 = Path.cwd().parent / "output" / "plot_6.png"
plt.savefig(path_to_save_6)
plt.show()


# Generating a Scatter plot of correlation prices of properties and Construction Year
filtered_data = data_prop[data_prop['constructionYear'] != 'No_information']
sns.scatterplot(data=filtered_data, x='constructionYear',
                y='price', color=custom_palette[5])
plt.ylim(0, 4e6)
plt.title('Prices of properties vs. Construction Year')
plt.xlabel('Construction Year')
plt.ylabel('Prices of properties, euros')
path_to_save_7 = Path.cwd().parent / "output" / "plot_7.png"
plt.savefig(path_to_save_7)
plt.show()


# Generating a Scatter plot of correlation prices of properties and Primary Energy Consumption
filtered_data = data_prop[data_prop['primaryEnergyConsumptionPerSqm']
                          != 'No_information']
sns.scatterplot(data=filtered_data, x='primaryEnergyConsumptionPerSqm',
                y='price', color=custom_palette[7])
plt.xlim(0, 2000)
plt.title('Prices of properties vs. Primary Energy Consumption')
plt.xlabel('Primary Energy Consamption, kWh/m²')
plt.ylabel('Prices of properties, euros')
path_to_save_8 = Path.cwd().parent / "output" / "plot_8.png"
plt.savefig(path_to_save_8)
plt.show()


# Generating a Bar Plot of Energy Class of properties in Belgium
sns.barplot(data=data_prop, x=data_prop['epcScore'].sort_values(
), y='price', estimator=lambda x: x.mean(), palette=custom_palette)
plt.xlabel('Energy Class of the properties')
plt.ylabel('Average prices of the properties, euros')
plt.title('Bar plot of average prices of properties depending on energy class')
plt.xticks(rotation=90)
path_to_save_9 = Path.cwd().parent / "output" / "plot_9.png"
plt.savefig(path_to_save_9)
plt.show()


# Generating a map of density of porperties for sale
m = folium.Map(location=(50.5039, 4.4699),
               tiles="cartodb positron", zoom_start=8)
path_to_save_95 = Path.cwd().parent / "output" / "footprint.html"
m.save(path_to_save_95)
data_map = data_prop[['id', 'latitude', 'longitude']].copy()
# Filter out rows with missing latitude or longitude information
filt_data_map = data_map[(data_map['latitude'] != 'No_information') & (
    data_map['longitude'] != 'No_information')]
locations = filt_data_map[['latitude', 'longitude']].values.tolist()
print(len(locations))
# Create a heatmap layer
heatmap = HeatMap(locations, radius=5, blur=3)
# Add the heatmap layer to the map
heatmap.add_to(m)
# Save the map with markers
path_to_save_10 = Path.cwd().parent / "output" / "footprint_with_markers.html"
m.save(path_to_save_10)
displayhook(m)


# Generating a map of density of apartments for sale
data_map = data_prop[['type', 'latitude', 'longitude']].copy()
# Filter out rows with missing latitude or longitude information
filt_apart_map = data_map[data_map['type'] == 'Apartment']
filt_apart_map = filt_apart_map[(filt_apart_map['latitude'] != 'No_information') & (
    filt_apart_map['longitude'] != 'No_information')]
# Create a list of latitudes and longitudes from the filtered data
locations_apart = filt_apart_map[['latitude', 'longitude']].values.tolist()
# Create a heatmap layer with prices as the weight
heatmap = HeatMap(locations_apart, radius=5, blur=3)
# Add the heatmap layer to the map
heatmap.add_to(m)
# Save the map with the heatmap and markers
path_to_save_11 = Path.cwd().parent / "output" / "footprint_with_heatmap_apart.html"
m.save(path_to_save_11)
# Display the map
displayhook(m)


# Generating a map of density of houses for sale
data_map = data_prop[['type', 'latitude', 'longitude']].copy()
# Filter out rows with missing latitude or longitude information
filt_houses_map = data_map[data_map['type'] == 'House']
filt_houses_map = filt_houses_map[(filt_houses_map['latitude'] != 'No_information') & (
    filt_houses_map['longitude'] != 'No_information')]
# Create a list of latitudes and longitudes from the filtered data
locations_houses = filt_houses_map[['latitude', 'longitude']].values.tolist()
# Create a heatmap layer with prices as the weight
heatmap = HeatMap(locations_houses, radius=5, blur=3)
# Add the heatmap layer to the map
heatmap.add_to(m)
# Save the map with the heatmap and markers
path_to_save_12 = Path.cwd().parent / "output" / "footprint_with_heatmap_houses.html"
m.save(path_to_save_12)
# Display the map
displayhook(m)
