# Brief:
EVs are the trend nowadays not just for the environmental factors but have also appeal visually to the customers. To understand the market size, the manufacturing numbers and the predicted numbers for the next 10 years, utilized the 'Detailed India EV Market Data' from Kaggle.

### Dataset Description: 
- EV Maker by Place: List of popular EV Makers and their location of Manufacturing Plant

- Operational PC: Total Operational Public Charging Station for EV available in each state

- Vehicle Class: Total vehicles (includes electric and all other fuels) registered (manufactured) by category from 2001 - Aug 2024

- ev_cat_01-24: Total electric vehicles manufactured from 2001 - Aug 2024 and vehicle category

- ev_sales_by_makers_and_cat_15-24: Total electric vehicles manufactured by makers from 2015 - Aug 2024 with the vehicle class 

### Data preprocessing and Cleaning:

After reading in the CSV files using pd.read_csv, created a dict with the top 10 rows of data for each of the above datasets.

```python
detail_overview = {
    'EV_maker' : EV_maker.head(10),
    'ev_cat' : ev_cat.head(10),
    'ev_sales': ev_sales.head(10),
    'Operations_df': Operations_df.head(10),
    'Vehicle_class': Vehicle_class.head(10)
}
```

```python
Vehicle_class.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 16 entries, 0 to 15
Data columns (total 2 columns):
 #   Column              Non-Null Count  Dtype 
---  ------              --------------  ----- 
 0   Vehicle Class       16 non-null     object
 1   Total Registration  16 non-null     object
dtypes: object(2)
memory usage: 384.0+ bytes
```
```python
ev_cat.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 284 entries, 0 to 283
Data columns (total 17 columns):
 #   Column                           Non-Null Count  Dtype 
---  ------                           --------------  ----- 
 0   Date                             284 non-null    object
 1   FOUR WHEELER (INVALID CARRIAGE)  284 non-null    int64 
 2   HEAVY GOODS VEHICLE              284 non-null    int64 
 3   HEAVY MOTOR VEHICLE              284 non-null    int64 
 4   HEAVY PASSENGER VEHICLE          284 non-null    int64 
 5   LIGHT GOODS VEHICLE              284 non-null    int64 
 6   LIGHT MOTOR VEHICLE              284 non-null    int64 
 7   LIGHT PASSENGER VEHICLE          284 non-null    int64 
 8   MEDIUM GOODS VEHICLE             284 non-null    int64 
 9   MEDIUM PASSENGER VEHICLE         284 non-null    int64 
 10  MEDIUM MOTOR VEHICLE             284 non-null    int64 
 11  OTHER THAN MENTIONED ABOVE       284 non-null    int64 
 12  THREE WHEELER(NT)                284 non-null    int64 
 13  TWO WHEELER (INVALID CARRIAGE)   284 non-null    int64 
 14  THREE WHEELER(T)                 284 non-null    int64 
 15  TWO WHEELER(NT)                  284 non-null    int64 
 16  TWO WHEELER(T)                   284 non-null    int64 
dtypes: int64(16), object(1)
memory usage: 37.8+ KB
```
All the incorrect data-types were corrected and processed for further analysis.

```python
fig, axes = plt.subplots(1,2, figsize = (30, 12))

axes[0].plot(ev_cat.groupby('Year').sum())

axes[0].set_xlim(2014,2025)



axes[1].plot(ev_cat.groupby('Year').sum())

axes[1].set_xlim(2020,2025)

fig.legend(ev_cat.groupby('Year').sum().columns,mode="expand",ncol=8)
plt.show()
```
![dowload](https://github.com/user-attachments/assets/66346482-f8e0-46a1-be87-c2e03622e6d3)

### Inference:
- Up until 2014-2015, the production count has not been that high to analyse so, considering the sales from 2014 from where there is a visible spike.
- Post 2020, there is some considerable increase in the manufacturing numbers of the vehicles mostly of Two wheelers(T and NT), Three Wheeler(T) and LMV
- The progression from 2021 to 2023 is a bigger spike with 2023-2024 seeing a slight decline.

```python
cleaned_data = (ev_cat.groupby('Year').sum().iloc[:].pct_change()*100).loc['2021':'2024'].replace([np.inf, np.nan], 0)

from matplotlib import cm
cmap = cm.get_cmap('Spectral')
cleaned_data.plot(kind='bar', stacked=True, figsize=(20, 12), cmap=cmap)
```

![download1](https://github.com/user-attachments/assets/9f11ea21-0dd6-4ba9-9c3c-932dbb576561)

### Inference:
- Light Goods Vehicle had the highest percentage increase and 2021 had a larger variance in terms of the number of sales.
- 2024 was the worst with negative or a downgrade from the previous years.
- 2023 although had less variance, seemed to be the most stable increment and sales spread across the vehicle categories.

```python
sales_df = ev_sales.groupby('Maker').sum()

top_15_companies = sales_df.sort_values(by='2024', ascending=False).head(15)

top_15 = (top_15_companies.pct_change(axis='columns')) * 100
top_15.replace([np.nan, np.inf], 0, inplace=True)


fig, axes = plt.subplots(1,3, figsize = (12, 8))

sns.barplot(x=top_15.index, y =top_15['2024']
           , ax = axes[0])


sns.barplot(x=top_15.index, y =top_15['2023']
           , ax = axes[1])

sns.barplot(x=top_15.index, y =top_15['2022']
           , ax = axes[2])

for ax in axes:
    ax.tick_params(axis='x',labelrotation=90)
    
plt.tight_layout(pad = 2);
```
![download3](https://github.com/user-attachments/assets/681d1ae7-73bb-4f85-ab59-0a5073931af1)

###Inference: 
The top 15 companies by 2024 have had no consistency in any of the past 3 years with majority of them making a negative or negligible sales in 2024.

1. ola electric technologies pvt ltd
1. tvs motor company ltd
1. bajaj auto ltd
1. ather energy pvt ltd
1. tata passenger electric mobility ltd
1. mahindra last mile mobility ltd
1. yc electric vehicle
1. hero motocorp ltd
1. greaves electric mobility pvt ltd
1. saera electric auto pvt ltd
1. piaggio vehicles pvt ltd
1. dilli electric auto pvt ltd
1. bgauss auto private limited
1. kinetic green energy & power solutions ltd
1. unique international

- 2024, majority of the makers had a negative percent change in sales from previous year(2023).
- 2023, has been the most profitable or successful year of EV's for all manufacturers combined.
- 2022 was the most positive percent increase which implies that 2021 was the year where sales had a positive impact throughout.

```python
df = ((ev_cat.groupby('Year').sum().iloc[:].sum()).to_frame().reset_index())
df.rename(columns = {'index': 'Vehicle Class', 0 : 'EV Registration'}, inplace =True)

plt.figure(figsize=(12,9))
plt.pie(Vehicle_class['Total Registration'], labels = Vehicle_class['Vehicle Class'], data = Vehicle_class
       ,radius=1.5, rotatelabels=True);

plt.legend(loc='upper left', bbox_to_anchor=(1,0.25))
```
![download3](https://github.com/user-attachments/assets/cf71dbab-8037-4d44-b53b-da1fe25664b1)

### Inference:
TWO WHEELER(NT)	and LMV have the highest registrations/manufacturing numbers of all the Vehicle class.

```python
Registrations = df.merge(Vehicle_class, on = 'Vehicle Class')

Vehicle Class	EV Registration	Total Registration	Percent share
0	FOUR WHEELER (INVALID CARRIAGE)	97	21346	0.454418
1	HEAVY GOODS VEHICLE	614	5870865	0.010458
2	HEAVY MOTOR VEHICLE	146	102965	0.141796
3	HEAVY PASSENGER VEHICLE	8131	828189	0.981781
4	LIGHT GOODS VEHICLE	12725	10249591	0.124151
5	LIGHT MOTOR VEHICLE	180330	65061773	0.277167
6	LIGHT PASSENGER VEHICLE	22807	4343410	0.525094
7	MEDIUM GOODS VEHICLE	31	875789	0.003540
8	MEDIUM PASSENGER VEHICLE	776	325015	0.238758
9	MEDIUM MOTOR VEHICLE	49	194600	0.025180
10	OTHER THAN MENTIONED ABOVE	7589	1126398	0.673741
11	THREE WHEELER(NT)	1498	679804	0.220358
12	TWO WHEELER (INVALID CARRIAGE)	126	110788	0.113731
13	THREE WHEELER(T)	1932683	10708473	18.048166
14	TWO WHEELER(NT)	2308887	274971646	0.839682
15	TWO WHEELER(T)	13459	129181	10.418715

```
### Inference:
- The Two wheeler(NT) as previously stated has the highest EV registrations with 3 wheelers(T) following.
- Three wheeler(T) however, as a category has the highest percentage of EVs manufactured/registered to-date followed by TWO WHEELER(T)

## Prediction of manufacturing numbers and the vehicle class
1. Looking at the numbers, EV market can be capitalized by increasing the production numbers of TWO WHEELER(NT)

1. With increasing LMVs, this category could be utilized as well to expand the market horizon.

1. Other categories with focus should be TWO WHEELER(T) and THREE WHEELER(T).

### Predictions for LMV
```python
category_specific = ev_cat.groupby('Year').sum()[['LIGHT GOODS VEHICLE', 'LIGHT MOTOR VEHICLE', 'THREE WHEELER(T)','TWO WHEELER(NT)', 'TWO WHEELER(T)']]


# 2024 has data only till August which is only a partial information so, considering complete data year
category_specific['LIGHT MOTOR VEHICLE'].index <= 2023

from scipy.optimize import curve_fit

# filter the dataset to include years with complete data, assuming 2023 is the last complete year
filtered_years = category_specific['LIGHT MOTOR VEHICLE'][category_specific['LIGHT MOTOR VEHICLE'].index <= 2023]

# define a function for exponential growth to fit the data
def exp_growth(x, a, b):
    return a * np.exp(b * x)

# prepare the data for curve fitting
x_data = filtered_years.index - filtered_years.index.min()
y_data = filtered_years.values

# fit the data to the exponential growth function
params, covariance = curve_fit(exp_growth, x_data, y_data)

# use the fitted function to forecast the number of EVs for 2024 and the next ten years
forecast_years = np.arange(2024, 2024 + 10) - filtered_years.index.min()
forecasted_values = exp_growth(forecast_years, *params)

# create a dictionary to display the forecasted values for easier interpretation
forecasted_evs = dict(zip(forecast_years + filtered_years.index.min(), forecasted_values))

print(forecasted_evs)

{2024: 175297.2235062509, 2025: 416174.7357889515, 2026: 988044.2327874522, 2027: 2345724.8169903485, 2028: 5569006.66432824, 2029: 13221429.471477449, 2030: 31389116.19354994, 2031: 74521186.80039199, 2032: 176921428.685528, 2033: 420030776.10626346}


# prepare data for plotting
years = np.arange(filtered_years.index.min(), 2033 + 1)
actual_years = filtered_years.index
forecast_years_full = np.arange(2024, 2033 + 1)

# actual and forecasted values
actual_values = filtered_years.values
forecasted_values_full = [forecasted_evs[year] for year in forecast_years_full]

plt.figure(figsize=(12, 8))
plt.plot(actual_years, actual_values, 'bo-', label='Actual Registrations')
plt.plot(forecast_years_full, forecasted_values_full, 'ro--', label='Forecasted Registrations')

plt.title('Current & Estimated LMV EV Market')
plt.xlabel('Year')
plt.ylabel('Number of EV Registrations')
plt.legend()
plt.grid(True)

plt.show()
```
![download](https://github.com/user-attachments/assets/b2bd75e4-96fd-45c5-9dee-19990adb00de)


### Predictions for 2W(NT)
```python
filtered_years_2w = category_specific['TWO WHEELER(NT)'][category_specific['TWO WHEELER(NT)'].index <= 2023]

# prepare the data for curve fitting
x_data = filtered_years_2w.index - filtered_years_2w.index.min()
y_data = filtered_years_2w.values

# fit the data to the exponential growth function
params, covariance = curve_fit(exp_growth, x_data, y_data)

# use the fitted function to forecast the number of EVs for 2024 and the next five years
forecast_years_2w = np.arange(2024, 2024 + 10) - filtered_years_2w.index.min()
forecasted_values_2w = exp_growth(forecast_years_2w, *params)

# create a dictionary to display the forecasted values for easier interpretation
forecasted_evs_2w = dict(zip(forecast_years_2w + filtered_years_2w.index.min(), forecasted_values_2w))

print(forecasted_evs_2w)

{2024: 1804738.3513997972, 2025: 3580417.2697123555, 2026: 7103183.580773072, 2027: 14091993.525161844, 2028: 27957081.39245217, 2029: 55464005.04574611, 2030: 110034942.93739386, 2031: 218298131.50437358, 2032: 433081282.6014201, 2033: 859189201.6993011}


years = np.arange(filtered_years_2w.index.min(), 2033 + 1)
actual_years = filtered_years_2w.index
forecast_years_full = np.arange(2024, 2033 + 1)

# actual and forecasted values
actual_values = filtered_years_2w.values
forecasted_values_full = [forecasted_evs_2w[year] for year in forecast_years_full]

plt.figure(figsize=(12, 8))
plt.plot(actual_years, actual_values, 'bo-', label='Actual Registrations')
plt.plot(forecast_years_full, forecasted_values_full, 'ro--', label='Forecasted Registrations')

plt.title('Current & Estimated 2W(NT) EV Market')
plt.xlabel('Year')
plt.ylabel('Number of EV Registrations')
plt.legend()
plt.grid(True)

plt.show()
```
![download (1)](https://github.com/user-attachments/assets/46432777-1606-4e7e-90d8-fe49ef691689)


### Predictions for 3W(T)
```python
filtered_years_3w = category_specific['THREE WHEELER(T)'][category_specific['THREE WHEELER(T)'].index <= 2023]

# prepare the data for curve fitting
x_data = filtered_years_3w.index - filtered_years.index.min()
y_data = filtered_years_3w.values

# fit the data to the exponential growth function
params, covariance = curve_fit(exp_growth, x_data, y_data)

# use the fitted function to forecast the number of EVs for 2024 and the next five years
forecast_years_3w = np.arange(2024, 2024 + 10) - filtered_years_3w.index.min()
forecasted_values_3w = exp_growth(forecast_years_3w, *params)

# create a dictionary to display the forecasted values for easier interpretation
forecasted_evs_3w = dict(zip(forecast_years_3w + filtered_years_3w.index.min(), forecasted_values_3w))


years = np.arange(filtered_years_3w.index.min(), 2033 + 1)
actual_years = filtered_years_3w.index
forecast_years_full = np.arange(2024, 2033 + 1)

# actual and forecasted values
actual_values = filtered_years_3w.values
forecasted_values_full = [forecasted_evs_3w[year] for year in forecast_years_full]

plt.figure(figsize=(12, 8))
plt.plot(actual_years, actual_values, 'bo-', label='Actual Registrations')
plt.plot(forecast_years_full, forecasted_values_full, 'ro--', label='Forecasted Registrations')

plt.title('Current & Estimated 2W(NT) EV Market')
plt.xlabel('Year')
plt.ylabel('Number of EV Registrations')
plt.legend()
plt.grid(True)

plt.show()
```
![download (2)](https://github.com/user-attachments/assets/955589dd-08ca-4e94-ae6c-d362a5653e0a)
