import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
from scipy import stats


from typing import Optional, Sequence

class InventoryImpairment:
	"""
	Inventory Impairment model class to predict the inventory impairment for a given stock, based on the past two years of data.
	"""		
	def __init__(self, random_state: int = 42) -> None:
		np.random.seed(random_state)
		tf.random.set_seed(random_state)
		self.random_state = random_state

	def get_monthly_data(self, data):
		"""
		"""
		self.months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

		for month_idx, month in enumerate(self.months):
			data[f'{month}_2022'] = 0
			data[f'{month}_2023'] = 0

		for idx, row in data.iterrows():
			year = pd.to_datetime(row[self.last_exit_date_variable]).year
			if year == self.predict_year:
				month_first_year = 12
				month_second_year = 12
			elif year == self.second_year:
				month_first_year = 12
				month_second_year = pd.to_datetime(row[self.last_exit_date_variable]).month
			elif year == self.first_year:
				month_first_year = pd.to_datetime(row[self.last_exit_date_variable]).month
				month_second_year = 0
			elif year < self.first_year:
				month_first_year = 0
				month_second_year = 0
			else:
				month_first_year = 12
				month_second_year = 12

			sales_first_year = row[f"{self.sales_variable_prefix}_{self.first_year}"] if not pd.isna(row[f"{self.sales_variable_prefix}_{self.first_year}"]) else 0
			sales_second_year = row[f"{self.sales_variable_prefix}_{self.second_year}"] if not pd.isna(row[f"{self.sales_variable_prefix}_{self.second_year}"]) else 0

			values_first_year, values_second_year = self.get_monthly_values(last_month_first_year=month_first_year, last_month_second_year=month_second_year, sales_first_year=sales_first_year, sales_second_year=sales_second_year)

			for month_idx, month in enumerate(self.months):
				row[f'{month}_2022'] = values_first_year[month_idx] if month_idx < len(values_first_year) else 0
				row[f'{month}_2023'] = values_second_year[month_idx] if month_idx < len(values_second_year) else 0
			
			return data
	
	def get_monthly_values(self, last_month_first_year: int, last_month_second_year: int, sales_first_year: float, sales_second_year: float):
		"""
		"""
		num_months_in_year = 12

		# Calculate monthly average values
		monthly_avg_sales_first_year = sales_first_year / last_month_first_year if last_month_first_year != 0 else 0
		monthly_avg_sales_second_year = sales_second_year / last_month_second_year if last_month_second_year != 0 else 0

		# Generate monthly sales data
		monthly_avg_sales_first_year = self.generate_monthly_sales(total_year_sales=sales_first_year, monthly_avg_sales=monthly_avg_sales_first_year, last_month=last_month_first_year)
		monthly_avg_sales_second_year = self.generate_monthly_sales(total_year_sales=sales_second_year, monthly_avg_sales=monthly_avg_sales_second_year, last_month=last_month_second_year)

		# Adjust monthly sales to match yearly total
		monthly_sales_first_year = self.calculate_monthly_sales(total_year_sales=sales_first_year, monthly_avg_sales=monthly_avg_sales_first_year, last_month=last_month_first_year)
		monthly_sales_second_year = self.calculate_monthly_sales(total_year_sales=sales_second_year, monthly_avg_sales=monthly_avg_sales_second_year, last_month=last_month_second_year)

		if last_month_first_year == 0:
			last_month_first_year = num_months_in_year
		if last_month_second_year == 0:
			last_month_second_year = num_months_in_year

		return monthly_sales_first_year, monthly_sales_second_year

	def generate_monthly_sales(self, monthly_avg_sales: float, last_month: int):
		# Generate monthly sales data
		monthly_sales = np.random.normal(loc=monthly_avg_sales, scale=monthly_avg_sales * self.variability, size=last_month)

		# Apply seasonal weights
		monthly_sales = monthly_sales * self.monthly_weights

		return monthly_sales

	def calculate_monthly_sales(self, total_year_sales: float, monthly_avg_sales: list[float], last_month: int):
		# Calculate surplus/deficit
		surplus_deficit = (total_year_sales - np.sum(monthly_avg_sales)) / last_month
		monthly_sales = monthly_avg_sales + surplus_deficit

		while np.min(monthly_sales) < 0:
			# Adjust monthly data to match yearly total
			monthly_sales = monthly_avg_sales + surplus_deficit
			monthly_sales = np.maximum(monthly_sales, 0)
			surplus_deficit = total_year_sales - np.sum(monthly_sales)

		return monthly_sales
	
	def fit_auto_arima_and_forecast(self, series):
		model = auto_arima(series, m=4)
		forecast = model.predict(n_periods=12)

		return forecast
	
	def create_auto_arima_and_forecast(self, data):
		ts = [f"{month}_{year}" for year in [self.first_year, self.second_year] for month in self.months]
		data.fillna(0, inplace=True)

		self.arima_forecasts = {}
		for product in data[self.id_variable]:
			product_sales = data[data[self.id_variable] == product][ts].values.flatten()
			forecast = self.fit_auto_arima_and_forecast(series=product_sales)
			self.arima_forecasts[product] = forecast

		sum_product_forecasts = [np.sum(forecast) for forecast in self.arima_forecasts.values()]

		indicators = self.calculate_decrease(current_sales=data[f"{self.sales_variable_prefix}_{self.second_year}"], predicted_sales=sum_product_forecasts)

		scaler = MinMaxScaler()
		indicators_by_column = scaler.fit_transform([[i] for i in indicators])
		
		indicators = [i[0] for i in indicators_by_column]

		return pd.Series(indicators)

	def calculate_decrease(self, current_sales, predicted_sales):
		decrease_indicator = []

		for current, predicted in zip(current_sales, predicted_sales):
			if predicted <= 0:
				decrease_indicator.append(float('-inf'))
			else:
				decrease_percentage = - ((current - predicted) / current) * 100 if current != 0 else -10
				decrease_indicator.append(decrease_percentage)

		# Replace infinite values with the maximum value
		max_decrease = max(decrease_indicator)
		decrease_indicator = [max_decrease if value == float('-inf') else value for value in decrease_indicator]
		
		return decrease_indicator	
	
	def create_auto_encoder(self, data, auto_arima_indicators):
		# New variables
		data['forecast_index'] = auto_arima_indicators
		data['cost_value'] = data[f"{self.unitary_sale_price_variable_prefix}_{self.second_year}"] / data[f"{self.unitary_cost_stock_variable_prefix}_{self.second_year}"]
		data['proportion_sales_stock'] = data[f"{self.number_of_units_sold_variable_prefix}_{self.second_year}"] / data[f"{self.quantity_stock_variable_prefix}_{self.second_year}"]

		# Drop rows with missing values
		data.dropna(inplace=True)

		# Select only relevant featuress
		features = [
			self.proportion_variation_unitary_sale_price_firstyear_secondyear_variable,
			self.difference_entry_exit_variable,
			self.last_exit_days_variable, 
			'proportion_sales_stock', 
			'cost_value', 
			'forecast_index'
		]

		X = data[features]

		# Split data into train and test sets
		X_train, X_val = train_test_split(X, test_size=0.2, random_state=self.random_state)

		# Define autoencoder architecture
		input_dim = X_train.shape[1]
		encoding_dim = 4

		input_layer = Input(shape=(input_dim,))
		encoder = Dense(128, activation='relu')(input_layer)
		encoder = Dense(encoding_dim, activation='relu')(encoder)
		decoder = Dense(128, activation='relu')(encoder)
		decoder = Dense(input_dim, activation='relu')(decoder)

		autoencoder = Model(inputs=input_layer, outputs=decoder)
		autoencoder.compile(optimizer='adam', loss='mse')

		# Train the Autoencoder model
		autoencoder.fit(X_train, X_train,
						epochs=50,
						batch_size=32,
						shuffle=True,
						validation_data=(X_val, X_val))
		
		# Create fake data of values that would need depreciation
		data_new = {
			self.proportion_variation_unitary_sale_price_firstyear_secondyear_variable: [-2],
			self.difference_entry_exit_variable: [600],
			self.last_exit_days_variable: [500],
			'proportion_sales_stock' : [0.1],
			'cost_value' : [0.1],
			'forecast_index' : [1]}
		new_dataframe = pd.DataFrame(data_new)

		# Create the model
		autoencoder = Model(inputs=input_layer, outputs=encoder)

		references = autoencoder.predict(new_dataframe)

		embeddings = autoencoder.predict(X)

		reference = references[0]

		distances = np.linalg.norm(embeddings - reference, axis=1)
		scaler = MinMaxScaler()
		distances = scaler.fit_transform([[d] for d in distances])
		
		distances = [1-d[0] for d in distances]

		return pd.Series(distances)
	
	def calculate_impairment_index_formula(self, data):
		# Calculate differences and ratios ensuring no division by zero
		data['delta_units'] = (data[f'{self.number_of_units_sold_variable_prefix}_{self.second_year}'] - data[f'{self.number_of_units_sold_variable_prefix}_{self.first_year}']) / (data[f'{self.number_of_units_sold_variable_prefix}_{self.first_year}'] + 1)
		data['delta_unitary_sell_price'] = data[f'{self.variation_unitary_sale_price_firstyear_secondyear_variable}_{self.first_year}_{self.second_year}'] / (data[f'{self.unitary_sale_price_variable_prefix}_{self.first_year}'] + 0.01)
		data['inventory_rotation'] = data[f'{self.number_of_units_sold_variable_prefix}_{self.second_year}'] / (data[f'{self.quantity_stock_variable_prefix}_{self.second_year}'] + 1)
		data['inactivity'] = data[self.last_exit_days_variable] + data[self.last_entry_date_variable]

		# Impairment index
		data['impairment_index'] = (
			-0.3 * data['delta_units'] +
			-0.2 * data['delta_unitary_sell_price'] +
			0.2 * np.log1p(data['inactivity']) +
			-0.3 * data['inventory_rotation']
		)
		
		# Ensure no NaN values
		data['impairment_index'].fillna(0, inplace=True)

		return data
	
	def indexs_interpretation(self, data, impairment_index, autoencoder_indexs, auto_arima_indexs):
		impairment_index = [i if i > 0 else 0 for i in impairment_index]

		scaler = MinMaxScaler()
		impairment_index = scaler.fit_transform([[i] for i in impairment_index])
		impairment_index = [i[0] for i in impairment_index]

		data['auto_arima_index'] = auto_arima_indexs
		data['autoencoder_index'] = autoencoder_indexs
		data['impairment_index'] = impairment_index
		data['merged_indexs'] = auto_arima_indexs + impairment_index + autoencoder_indexs


		# Mode of the discretised values (by round 2)
		mode = [round(v, 2) for v in stats.mode(data['merged_indexs'])[0][0]]
		data['fair_price'] = data[f'{self.unitary_sale_price_variable_prefix}_{self.second_year}'] - data[f'{self.unitary_cost_stock_variable_prefix}_{self.second_year}' * (data['merged_indexs'] - mode/self.tolerance)]

		data["new_value"] = data[["fair_price", f"{self.unitary_cost_stock_variable_prefix}_{self.second_year}"]].min(axis=1)

		return data
	
	def explain_indexs(self):
		"""
		"""
		pass

	# CALLABLE METHODS

	def fit(self, 
		data, 
		id_variable: str = 'material', 
		stock_ids: Optional[Sequence[str]] = None,
		monthly_weights: list[float] = [0.8, 1, 1, 1, 1, 1.2, 1.3, 1.1, 1, 1, 1, 1],
		last_exit_date_variable: str = "data_darrera_entrada",
		last_exit_days_variable: str = "dies_ultima_sortida",
		last_entry_date_variable: str = "data_darrera_entrada",
		difference_entry_exit_variable: str = "diferencia_entrada_sortida",
		proportion_variation_unitary_sale_price_firstyear_secondyear_variable: Optional[str] = None,
		variation_unitary_sale_price_firstyear_secondyear_variable: Optional[str] = None,
		sales_variable_prefix: str = "vendes",
		unitary_sale_price_variable_prefix: str = "preu_venda_unitari",
		unitary_cost_stock_variable_prefix: str = "cost_unitari_stock",
		number_of_units_sold_variable_prefix: str = "unitats",
		quantity_stock_variable_prefix: str = "stock_final",
		first_year: int = 2022,
		second_year: int = 2023,
		predict_year: int = 2024,
		variability: float = 0.0,
		):
		"""
		Fit the model to the data provided.

		Parameters
		----------

		Returns
		-------
		None
		"""
		assert variability >= 0, "Variability must be greater or equal to 0."
		# Init all variables
		if proportion_variation_unitary_sale_price_firstyear_secondyear_variable is None:
			print(f"No proportion_variation_unitary_sale_price_firstyear_secondyear_variable provided. Using proporcio_variacio_preu_venda_unitari_{first_year}_{second_year} instead.")
			proportion_variation_unitary_sale_price_firstyear_secondyear_variable = f"proporcio_variacio_preu_venda_unitari_{first_year}_{second_year}"

		if variation_unitary_sale_price_firstyear_secondyear_variable is None:
			print(f"No variation_unitary_sale_price_firstyear_secondyear_variable provided. Using variacio_preu_venda_unitari_{first_year}_{second_year} instead.")
			variation_unitary_sale_price_firstyear_secondyear_variable = f"variacio_preu_venda_unitari_{first_year}_{second_year}"
		
		self.data = data
		self.last_exit_date_variable = last_exit_date_variable
		self.last_exit_days_variable = last_exit_days_variable
		self.last_entry_date_variable = last_entry_date_variable
		self.difference_entry_exit_variable = difference_entry_exit_variable
		self.first_year = first_year
		self.second_year = second_year
		self.predict_year = predict_year
		self.sales_variable_prefix = sales_variable_prefix
		self.variability = variability
		self.unitary_sale_price_variable_prefix = unitary_sale_price_variable_prefix
		self.unitary_cost_stock_variable_prefix = unitary_cost_stock_variable_prefix
		self.number_of_units_sold_variable_prefix = number_of_units_sold_variable_prefix
		self.quantity_stock_variable_prefix = quantity_stock_variable_prefix
		self.proportion_variation_unitary_sale_price_firstyear_secondyear_variable = proportion_variation_unitary_sale_price_firstyear_secondyear_variable
		self.variation_unitary_sale_price_firstyear_secondyear_variable = variation_unitary_sale_price_firstyear_secondyear_variable

		if stock_ids is None:
			stock_ids = self.data[id_variable].unique()
		
		self.monthly_weights = monthly_weights
		self.id_variable = id_variable

		# Get only the rows that are in the stock_ids
		self.stock_data = self.data[self.data[self.id_variable].isin(stock_ids)]

		# Get the monthly data
		self.data_monthly = self.get_monthly_data(data=self.stock_data)

		# Create auto arima model and forecast for each stock
		self.auto_arima_indexs = self.create_auto_arima_and_forecast(data=self.data_monthly)

		# Create auto encoder model
		self.autoencoder_indexs = self.create_auto_encoder(data=self.data, auto_arima_indicators=self.auto_arima_indexs)

		# Calculate impairment index
		self.impairment_indexs = self.calculate_impairment_index_formula(data=self.data)

	def predict(self, tolerance: float = 1.5):
		assert tolerance > 0, "Tolerance must be greater than 0."

		# Interpret the indexes
		self.tolerance = tolerance
		self.data_indexs_interpreted = self.indexs_interpretation(data=self.data, impairment_index=self.impairment_indexs, autoencoder_indexs=self.autoencoder_indexs, auto_arima_indexs=self.auto_arima_indexs)

		return self.data_indexs_interpreted['fair_price']
	
	def explain(self):
		"""
		"""
		pass

