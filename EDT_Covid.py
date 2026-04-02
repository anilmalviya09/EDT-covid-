import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
	import matplotlib.pyplot as plt
	_plotting_available = True
except Exception:
	_plotting_available = False


df = pd.read_csv('covid_data.csv')
    
# normalize column names from the CSV to the names expected by the script
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
cols_map = {}
if 'cumulative_cases' in df.columns:
	cols_map['cumulative_cases'] = 'total_cases'
if 'cumulative_deaths' in df.columns:
	cols_map['cumulative_deaths'] = 'total_deaths'
if cols_map:
	df = df.rename(columns=cols_map)

# ensure we have a `date` column and parse it
if 'date' in df.columns:
	df['date'] = pd.to_datetime(df['date'])
elif 'datetime' in df.columns:
	df['date'] = pd.to_datetime(df['datetime'])
else:
	raise KeyError('No date/datetime column found in covid_data.csv')

print(df.head())
print(df.info())

df['7_day_avg'] = df.groupby('country')['new_cases'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

df['cfr_pct'] = (df['total_deaths'] / df['total_cases']) * 100

print(df[['date', 'country', 'new_cases', '7_day_avg']].head(10))

peak_days = df.loc[df.groupby('country')['new_cases'].idxmax()]
print('--- Peak Infection Days by Country ---')
print(peak_days[['country', 'date', 'new_cases']])

avg_vac = df['vaccination_rate_pct'].mean()
print(f"\nGlobal Average Vaccination Rate: {avg_vac:.2f}%")

use_data = df[df['country'].str.upper() == 'USA'].sort_values('date')

if _plotting_available:
	plt.figure(figsize=(12, 6))
	plt.bar(use_data['date'], use_data['new_cases'], color='skyblue', alpha=0.4, label='Daily New Cases')
	plt.plot(use_data['date'], use_data['7_day_avg'], color='red', linewidth=2, label='7-Day Average')

	plt.title('COVID-19 trends in the USA (Synthetic Data)')
	plt.xlabel('Date')
	plt.ylabel('Number of Cases')
	plt.legend()
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	out_file = 'usa_covid_trends.png'
	plt.tight_layout()
	plt.savefig(out_file)
	print(f"Saved plot to {out_file}")
else:
	print('matplotlib not available — skipping plot (install matplotlib to enable).')

summary = df.groupby('country').agg({
    'total_cases': 'max',
    'total_deaths': 'max',
    'vaccination_rate_pct': 'max'
}).sort_values(by='total_cases', ascending=False)

print("\n--- Final Project Summary Table ---")
print(summary)