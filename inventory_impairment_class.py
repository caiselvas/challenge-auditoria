import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import warnings
from scipy.stats import mstats
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
from scipy import stats
from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show
import json

from typing import Optional, Sequence

class InventoryImpairment:
	"""
	Inventory Impairment model class to predict the inventory impairment for a given stock, based on the past two years of data.
	"""		
	def __init__(self, random_state: int = 42, forecast_file: str = './forecast/arima.json') -> None:
		np.random.seed(random_state)
		tf.random.set_seed(random_state)
		self.random_state = random_state
		self.forecast_file = forecast_file

	def get_monthly_data(self, data):
		data = data.copy()
		self.months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

		for month_idx, month in enumerate(self.months):
			data[f'{month}_2022'] = 0
			data[f'{month}_2023'] = 0

		for idx, row in data.iterrows():
			year = pd.to_datetime(row[self.last_exit_date_variable]).year
			if year == self.second_year + 1:
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
			data.loc[idx] = row
		
		return data
	
	def get_monthly_values(self, last_month_first_year: int, last_month_second_year: int, sales_first_year: float, sales_second_year: float):
		num_months_in_year = 12

		# Calculate monthly average values
		monthly_avg_sales_first_year = sales_first_year / last_month_first_year if last_month_first_year != 0 else 0
		monthly_avg_sales_second_year = sales_second_year / last_month_second_year if last_month_second_year != 0 else 0

		value_monthly_avg_sales_first_year = monthly_avg_sales_first_year

		# Generate monthly sales data
		monthly_avg_sales_first_year = self.generate_monthly_sales(monthly_avg_sales=monthly_avg_sales_first_year, last_month=last_month_first_year)
		monthly_avg_sales_second_year = self.generate_monthly_sales(monthly_avg_sales=monthly_avg_sales_second_year, last_month=last_month_second_year, monthly_avg_sales_previous = value_monthly_avg_sales_first_year)

		# Adjust monthly sales to match yearly total
		monthly_sales_first_year = self.calculate_monthly_sales(total_year_sales=sales_first_year, monthly_avg_sales=monthly_avg_sales_first_year, last_month=last_month_first_year)
		monthly_sales_second_year = self.calculate_monthly_sales(total_year_sales=sales_second_year, monthly_avg_sales=monthly_avg_sales_second_year, last_month=last_month_second_year)

		if last_month_first_year == 0:
			last_month_first_year = num_months_in_year
		if last_month_second_year == 0:
			last_month_second_year = num_months_in_year

		return monthly_sales_first_year, monthly_sales_second_year

	def generate_monthly_sales(self, monthly_avg_sales: float, last_month: int, monthly_avg_sales_previous: Optional[float] = None):
		# Generate monthly sales data 
		
		if monthly_avg_sales_previous != None:
			monthly_sales = np.linspace(monthly_avg_sales_previous, monthly_avg_sales, last_month)
			monthly_sales += np.random.normal(loc=0, scale=monthly_avg_sales * self.variability, size=last_month)

		else:
			monthly_sales = np.random.normal(loc=monthly_avg_sales, scale=monthly_avg_sales * self.variability, size=last_month)

		if len(monthly_sales) < 12:
			# Pad with zeros
			monthly_sales = np.pad(monthly_sales, (0, 12 - len(monthly_sales)), mode='constant', constant_values=0)
		
		# Apply monthly weights
		monthly_sales = np.array(monthly_sales) * np.array(self.monthly_weights)

		return monthly_sales

	def calculate_monthly_sales(self, total_year_sales: float, monthly_avg_sales: list[float], last_month: int):
		# Calculate surplus/deficit
		surplus_deficit = (total_year_sales - np.sum(monthly_avg_sales)) / last_month if last_month != 0 else 0
		monthly_sales = monthly_avg_sales + surplus_deficit

		while np.min(monthly_sales) < 0:
			# Adjust monthly data to match yearly total
			monthly_sales[last_month-1] = monthly_avg_sales[last_month-1] + surplus_deficit
			monthly_sales = np.maximum(monthly_sales, 0)
			surplus_deficit = total_year_sales - np.sum(monthly_sales)

		return monthly_sales
	
	def fit_auto_arima_and_forecast(self, series):
		model = auto_arima(series, m=4)
		forecast = model.predict(n_periods=12)

		return forecast
	
	def create_auto_arima_and_forecast(self, data):
		data = data.copy()
		ts = [f"{month}_{year}" for year in [self.first_year, self.second_year] for month in self.months]

		data.fillna(0, inplace=True)

		self.arima_forecasts = {}
		try:
			with open(self.forecast_file, "r") as json_file:
				self.arima_forecasts = json.load( json_file)
		except:
			print("Couldn't open file. Fitting auto_arima model.")
			for product in data[self.id_variable]:
				product_sales = data[data[self.id_variable] == product][ts].values.flatten()
				forecast = self.fit_auto_arima_and_forecast(series=product_sales)
				self.arima_forecasts[product] = forecast
			new_forecasts = {key: list(value) for key, value in self.arima_forecasts.items()}
			with open(self.forecast_file, "w") as json_file:
				json.dump(new_forecasts, json_file)


		sum_product_forecasts = [np.sum(forecast) for forecast in self.arima_forecasts.values()]

		indicators = self.calculate_decrease(current_sales=data[f"{self.sales_variable_prefix}_{self.second_year}"], predicted_sales=sum_product_forecasts)

		indicators = mstats.winsorize(np.array(indicators),  limits=[0.3, 0])
		scaler = MinMaxScaler()
		indicators_by_column = scaler.fit_transform([[i] for i in indicators])
		
		indicators = [i[0] for i in indicators_by_column]


		return pd.Series(indicators)

	def calculate_decrease(self, current_sales, predicted_sales):
		decrease_indicator = []

		for current, predicted in zip(current_sales, predicted_sales):
			if predicted <= 0:
				decrease_indicator.append(-100)
			else:
				decrease_percentage = - ((current - predicted) / current) * 100 if current != 0 else 0
				decrease_indicator.append(decrease_percentage)
		
		return decrease_indicator	
	
	def create_auto_encoder(self, data, auto_arima_indicators):
		data = data.copy()

		# New variables
		data['forecast_index'] = auto_arima_indicators.reset_index(drop = True)
		data['cost_value'] = data[f"{self.unitary_sale_price_variable_prefix}_{self.second_year}"] / data[f"{self.unitary_cost_stock_variable_prefix}_{self.second_year}"]
		data['proportion_sales_stock'] = data[f"{self.number_of_units_sold_variable_prefix}_{self.second_year}"] / data[f"{self.quantity_stock_variable_prefix}_{self.second_year}"]

		# Drop rows with missing values
		data = data.fillna(0)

		# Select only relevant featuress
		features = [
			self.proportion_variation_unitary_sale_price_firstyear_secondyear_variable,
			self.proportion_variation_units_firstyear_secondyear_variable,
			self.proportion_variation_sales_firstyear_secondyear_variable,
			self.difference_entry_exit_variable,
			self.last_exit_days_variable, 
			'proportion_sales_stock', 
			'cost_value', 
			'forecast_index'
		]

		
		X = data[features]

		scaler = StandardScaler()
		
		# Split data into train and test sets
		X_train, X_val = train_test_split(X, test_size=0.2, random_state=self.random_state)

		X_train = pd.DataFrame(scaler.fit_transform(X_train))
		X_val = pd.DataFrame(scaler.transform(X_val))

		# Define autoencoder architecture
		input_dim = X_train.shape[1]
		encoding_dim = 4

		input_layer = Input(shape=(input_dim,))
		encoder = Dense(128, activation='relu')(input_layer)
		encoder = Dense(64, activation='relu')(encoder)
		encoder = Dense(encoding_dim, activation='relu')(encoder)

		decoder = Dense(64, activation='relu')(encoder)
		decoder = Dense(128, activation='relu')(decoder)
		decoder = Dense(input_dim, activation='relu')(decoder)

		autoencoder = Model(inputs=input_layer, outputs=decoder)
		autoencoder.compile(optimizer="adam", loss='mse')

		# Train the Autoencoder model
		autoencoder.fit(X_train, X_train,
						epochs=70,
						batch_size=32,
						shuffle=True,
						validation_data=(X_val, X_val),
						verbose = 0)
		
		# Create fake data of values that would need depreciation
		data_new = {
			self.proportion_variation_unitary_sale_price_firstyear_secondyear_variable: [-100],
			self.proportion_variation_units_firstyear_secondyear_variable: [-100],
			self.proportion_variation_sales_firstyear_secondyear_variable: [-100],
			self.difference_entry_exit_variable: [730],
			self.last_exit_days_variable: [365],
			'proportion_sales_stock' : [0.01],
			'cost_value' : [0.01],
			'forecast_index' : [1]}

		new_dataframe = pd.DataFrame(data_new)
		# Create the model
		new_dataframe = pd.DataFrame(scaler.transform(new_dataframe))
		autoencoder = Model(inputs=input_layer, outputs=encoder)

		references = autoencoder.predict(new_dataframe, verbose=0)

		embeddings = autoencoder.predict(X, verbose=0)

		reference = references[0]

		if self.similarity == "dist_cos":
			cos_distances = np.abs(np.dot(embeddings, reference)) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(reference))

			eucl_distances = np.linalg.norm(embeddings - reference, axis=1)
			eucl_distances = mstats.winsorize(eucl_distances, limits=[0, 0.1])

			scaler = MinMaxScaler()
			cos_distances = scaler.fit_transform([[d] for d in cos_distances])
			eucl_distances = scaler.fit_transform([[d] for d in eucl_distances])

			eucl_distances = [1-d[0] for d in eucl_distances]
			cos_distances = [d[0] for d in cos_distances]

			distances = np.mean(np.array([eucl_distances, cos_distances]), axis=0)
			return pd.Series(distances)


		else:
			if self.similarity == "cos":
				distances = np.abs(np.dot(embeddings, reference)) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(reference))
			else:
				distances = np.linalg.norm(embeddings - reference, axis=1)

				distances = mstats.winsorize(distances, limits=[0, 0.1])

			scaler = MinMaxScaler()
			distances = scaler.fit_transform([[d] for d in distances])

			if self.similarity != "cos":
				distances = [1-d[0] for d in distances]
			else:
				distances = [d[0] for d in distances]

			return pd.Series(distances)
	
	def calculate_impairment_index_formula(self, data):
		data = data.copy()
		# Calculate differences and ratios ensuring no division by zero
		data['delta_units'] = (data[f'{self.number_of_units_sold_variable_prefix}_{self.second_year}'] - data[f'{self.number_of_units_sold_variable_prefix}_{self.first_year}']) / (data[f'{self.number_of_units_sold_variable_prefix}_{self.first_year}'] + 1)
		data['delta_unitary_sell_price'] = data[self.variation_unitary_sale_price_firstyear_secondyear_variable] / (data[f'{self.unitary_sale_price_variable_prefix}_{self.first_year}'] + 0.01)
		data['inventory_rotation'] = data[f'{self.number_of_units_sold_variable_prefix}_{self.second_year}'] / (data[f'{self.quantity_stock_variable_prefix}_{self.second_year}'] + 1)
		data['inactivity'] = data[self.last_exit_days_variable] + data[self.last_entry_days_variable]

		# Impairment index
		data['impairment_index'] = (
			self.impairment_index_coefficients[0] * data['delta_units'] +
			self.impairment_index_coefficients[1] * data['delta_unitary_sell_price'] +
			self.impairment_index_coefficients[2] * np.log1p(data['inactivity']) +
			self.impairment_index_coefficients[3] * data['inventory_rotation']
		)
		
		# Ensure no NaN values
		data['impairment_index'].fillna(0, inplace=True)

		# Keep only positive values
		data['impairment_index'] = data['impairment_index'].apply(lambda x: x if x > 0 else 0)

		# Scale the values of the impairment index
		scaler = MinMaxScaler()
		impairment_index_scaled = scaler.fit_transform(data['impairment_index'].values.reshape(-1, 1))
		# Replace the original column with the scaled values
		data['impairment_index'] = impairment_index_scaled.flatten()  # Flatten to convert it back to a 1D array


		return data['impairment_index']
	
	def indexs_interpretation(self, data, impairment_index, auto_encoder_indexs, auto_arima_indexs):
		data = data.copy()

		data.reset_index(inplace = True)

		# Aquest és pq sinó hi ha 15 fair prices que no es poden calcular
		data = data.fillna(0)

		data['impairment_index'] = impairment_index.reset_index(drop=True)
		data['auto_arima_index'] = auto_arima_indexs.reset_index(drop=True)
		data['auto_encoder_index'] = auto_encoder_indexs.reset_index(drop=True)
		data['merged_indexs'] = self.indexs_weights['auto_arima'] * data['auto_arima_index'] + self.indexs_weights['auto_encoder'] * data['auto_encoder_index'] + self.indexs_weights['impairment'] * data['impairment_index']

		# Mode of the discretised values (by round 2)
		rounded_merged_indexs = [round(v, 2) for v in data['merged_indexs']]
		values, counts = stats.mode(rounded_merged_indexs, keepdims=False)
		mode = values
		mean = data['merged_indexs'].mean()

		if isinstance(self.threshold, str):
			if self.threshold == 'mode':
				criterion = mode
				print(f"Mode used as threshold: {criterion}")
			elif self.threshold == 'mean':
				criterion = mean
				print(f"Mean used as threshold: {criterion}")
		else:
			criterion = self.threshold
			print(f"Value used as threshold: {criterion}")

		data['fair_price'] = data[f'{self.unitary_sale_price_variable_prefix}_{self.second_year}'] - data[f'{self.unitary_cost_stock_variable_prefix}_{self.second_year}'] * (data['merged_indexs'] - criterion/self.tolerance)

		data["new_value"] = data[["fair_price", f"{self.unitary_cost_stock_variable_prefix}_{self.second_year}"]].min(axis=1)
		return data

	def forecasts_to_excel(self, filepath: str):
		self.data_indexs_interpreted.to_excel(filepath, index=False)

	# CALLABLE METHODS
 
	def set_forecast_file(self, file):
		"""
		Set the file with the precalculated forecasts, to avoid recalculating them.

		Parameters
		----------
		file: str
			The path to the file with the forecasts.

		Returns
		--------
		None
		"""
		self.forecast_file = file

	def stock_management(self) -> None:
		"""
		Based on the predicted forecast, return an stock recommendation to better deal with shortages or excess. 
			Must have called fit and predict before.

		Parameters
		-----------
		None

		Returns
		--------
		None
		"""
		assert self.predicted, "Model must be predicted before calling stock_management."

		for product_id, sales_predictions in  self.arima_forecasts.items():
			total_projected_sales = sum(sales_predictions)

			quarter_projected_sales = sum(sales_predictions[:4])
			current_stock = self.data_indexs_interpreted.loc[self.data_indexs_interpreted[self.id_variable] == float(product_id), f"{self.quantity_stock_variable_prefix}_{self.second_year}"].values[0]
			fair_price = self.data_indexs_interpreted.loc[self.data_indexs_interpreted[self.id_variable] == float(product_id), 'fair_price'].values[0]

			total_projected_sales = total_projected_sales / fair_price # Unit projected sales
			quarter_projected_sales = quarter_projected_sales / fair_price
			
			# Determine if additional stock is needed or if there's excess inventory
			recommendation = False
			if total_projected_sales < current_stock:
				recommendation = f"Recommendation for {product_id}: Reduce stock. Projected sales: {total_projected_sales}, Current stock: {current_stock}, Fair price: {fair_price}"
			
			if total_projected_sales > current_stock:
				if quarter_projected_sales > current_stock:
					recommendation = f"Recommendation for {product_id}: Order additional stock (in a quatrimester you won't have any). Projected quatrimestral sales: {quarter_projected_sales}, Current stock: {current_stock}, Fair price: {fair_price}"

			if recommendation:
				print(recommendation)

	def fit(self, 
		data, 
		id_variable: str = 'material', 
		stock_ids: Optional[Sequence[str]] = None,
		monthly_weights: list[float] = [0.8, 1, 1, 1, 1, 1.2, 1.3, 1.1, 1, 1, 1, 1],
		impairment_index_coefficients: list[float] = [-0.3, -0.2, 0.2, -0.3],
		last_exit_date_variable: str = "data_darrera_entrada",
		last_exit_days_variable: str = "dies_ultima_sortida",
		last_entry_date_variable: str = "data_darrera_entrada",
		last_entry_days_variable: str = "dies_ultima_entrada",
		difference_entry_exit_variable: str = "diferencia_entrada_sortida",
		proportion_variation_unitary_sale_price_firstyear_secondyear_variable: Optional[str] = None,
		variation_unitary_sale_price_firstyear_secondyear_variable: Optional[str] = None,
		proportion_variation_units_firstyear_secondyear_variable: Optional[str] = None,
		variation_units_firstyear_secondyear_variable: Optional[str] = None,
		proportion_variation_sales_firstyear_secondyear_variable: Optional[str] = None,
		variation_sales_firstyear_secondyear_variable: Optional[str] = None,
		sales_variable_prefix: str = "vendes",
		unitary_sale_price_variable_prefix: str = "preu_venda_unitari",
		unitary_cost_stock_variable_prefix: str = "cost_unitari_stock",
		number_of_units_sold_variable_prefix: str = "unitats",
		quantity_stock_variable_prefix: str = "stock_final",
		total_value_stock_variable_prefix: str = "valor_total_stock",
		first_year: int = 2022,
		second_year: int = 2023,
		variability: float = 0.0,
		similarity: Optional[str] = 'dist'
		) -> None:
		"""
		Fit the model to the data provided.

		Parameters
		----------
		data: pd.DataFrame

		id_variable: str
			The name of the column that contains the stock ids.

		stock_ids: Optional[Sequence[str]]
			The ids present in the stock data. If None, the non NaN ids in with quantity_stock_variable_prefix_{second_year} are used.

		monthly_weights: list[float]
			The weights to apply to the monthly sales predictions.

		impairment_index_coefficients: list[float]
			The coefficients to apply to the different variables in the impairment index formula. The order is [delta_units, delta_unitary_sell_price, inactivity, inventory_rotation].

		last_exit_date_variable: str
			The name of the column that contains the last exit date.

		last_exit_days_variable: str
			The name of the column that contains the last exit days.

		last_entry_date_variable: str
			The name of the column that contains the last entry date.

		last_entry_days_variable: str
			The name of the column that contains the last entry days.

		difference_entry_exit_variable: str
			The name of the column that contains the difference between the entry and exit dates (in days).

		proportion_variation_unitary_sale_price_firstyear_secondyear_variable: Optional[str]
			The name of the column that contains the proportion of variation of the unitary sale price between the first and second year.

		variation_unitary_sale_price_firstyear_secondyear_variable: Optional[str]
			The name of the column that contains the variation of the unitary sale price between the first and second year.

		proportion_variation_units_firstyear_secondyear_variable: Optional[str]
			The name of the column that contains the proportion of variation of the units sold between the first and second year.

		variation_units_firstyear_secondyear_variable: Optional[str]
			The name of the column that contains the variation of the units sold between the first and second year.

		proportion_variation_sales_firstyear_secondyear_variable: Optional[str]
			The name of the column that contains the proportion of variation of the sales between the first and second year.
		
		variation_sales_firstyear_secondyear_variable: Optional[str]
			The name of the column that contains the variation of the sales between the first and second year.

		sales_variable_prefix: str
			The prefix of the sales variables. E.g. "vendes" for "vendes_2022" and "vendes_2023".

		unitary_sale_price_variable_prefix: str
			The prefix of the unitary sale price variables. E.g. "preu_venda_unitari" for "preu_venda_unitari_2022" and "preu_venda_unitari_2023".

		unitary_cost_stock_variable_prefix: str
			The prefix of the unitary cost stock variables. E.g. "cost_unitari_stock" for "cost_unitari_stock_2022" and "cost_unitari_stock_2023".

		number_of_units_sold_variable_prefix: str
			The prefix of the number of units sold variables (the units sold in a year). E.g. "unitats" for "unitats_2022" and "unitats_2023".

		quantity_stock_variable_prefix: str
			The prefix of the quantity stock variables (the units of a product remaining in the final stock of the second year). E.g. "stock_final" for "stock_final_2023".

		total_value_stock_variable_prefix: str
			The prefix of the total value stock variables (the total value of the stock in the second year). E.g. "valor_total_stock" for "valor_total_stock_2023".

		first_year: int
			The first year of the data.

		second_year: int
			The second year of the data (the year to predict).

		variability: float
			The variability of the sales data. The higher the variability, the more the sales will differ from the average.
		
		similarity: str
			The metric to use for distance calculation
			
			- "dist": Euclidean distance.
        	- "cos": Cosine similarity (converted to distance).

		Returns
		-------
		None
		"""
		assert variability >= 0, "Variability must be greater or equal to 0."
		assert len(impairment_index_coefficients) == 4, "Impairment index coefficients must have length 4 (delta_units, delta_unitary_sell_price, inactivity, inventory_rotation)."

		# Init all variables
		if proportion_variation_unitary_sale_price_firstyear_secondyear_variable is None:
			warnings.warn(f"No proportion_variation_unitary_sale_price_firstyear_secondyear_variable provided. Using proporcio_variacio_preu_venda_unitari_{first_year}_{second_year} instead.", UserWarning)
			proportion_variation_unitary_sale_price_firstyear_secondyear_variable = f"proporcio_variacio_preu_venda_unitari_{first_year}_{second_year}"

		if variation_unitary_sale_price_firstyear_secondyear_variable is None:
			warnings.warn(f"No variation_unitary_sale_price_firstyear_secondyear_variable provided. Using variacio_preu_venda_unitari_{first_year}_{second_year} instead.", UserWarning)
			variation_unitary_sale_price_firstyear_secondyear_variable = f"variacio_preu_venda_unitari_{first_year}_{second_year}"

		if proportion_variation_units_firstyear_secondyear_variable is None:
			warnings.warn(f"No proportion_variation_units_firstyear_secondyear_variable provided. Using proporcio_variacio_unitats_{first_year}_{second_year} instead.", UserWarning)
			proportion_variation_units_firstyear_secondyear_variable = f"proporcio_variacio_unitats_{first_year}_{second_year}"
		
		if variation_units_firstyear_secondyear_variable is None:
			warnings.warn(f"No variation_units_firstyear_secondyear_variable provided. Using variacio_unitats_{first_year}_{second_year} instead.", UserWarning)
			variation_units_firstyear_secondyear_variable = f"variacio_unitats_{first_year}_{second_year}"
		
		if proportion_variation_sales_firstyear_secondyear_variable is None:
			warnings.warn(f"No proportion_variation_sales_firstyear_secondyear_variable provided. Using proporcio_variacio_vendes_{first_year}_{second_year} instead.", UserWarning)
			proportion_variation_sales_firstyear_secondyear_variable = f"proporcio_variacio_vendes_{first_year}_{second_year}"

		if variation_sales_firstyear_secondyear_variable is None:
			warnings.warn(f"No variation_sales_firstyear_secondyear_variable provided. Using variacio_vendes_{first_year}_{second_year} instead.", UserWarning)
			variation_sales_firstyear_secondyear_variable = f"variacio_vendes_{first_year}_{second_year}"
		
		self.data = data
		self.impairment_index_coefficients = impairment_index_coefficients
		self.monthly_weights = monthly_weights
		self.id_variable = id_variable
		self.last_exit_date_variable = last_exit_date_variable
		self.last_exit_days_variable = last_exit_days_variable
		self.last_entry_date_variable = last_entry_date_variable
		self.last_entry_days_variable = last_entry_days_variable
		self.difference_entry_exit_variable = difference_entry_exit_variable
		self.first_year = first_year
		self.second_year = second_year
		self.sales_variable_prefix = sales_variable_prefix
		self.variability = variability
		self.unitary_sale_price_variable_prefix = unitary_sale_price_variable_prefix
		self.unitary_cost_stock_variable_prefix = unitary_cost_stock_variable_prefix
		self.number_of_units_sold_variable_prefix = number_of_units_sold_variable_prefix
		self.quantity_stock_variable_prefix = quantity_stock_variable_prefix
		self.total_value_stock_variable_prefix = total_value_stock_variable_prefix
		self.proportion_variation_unitary_sale_price_firstyear_secondyear_variable = proportion_variation_unitary_sale_price_firstyear_secondyear_variable
		self.variation_unitary_sale_price_firstyear_secondyear_variable = variation_unitary_sale_price_firstyear_secondyear_variable
		self.proportion_variation_units_firstyear_secondyear_variable = proportion_variation_units_firstyear_secondyear_variable
		self.variation_units_firstyear_secondyear_variable = variation_units_firstyear_secondyear_variable
		self.proportion_variation_sales_firstyear_secondyear_variable = proportion_variation_sales_firstyear_secondyear_variable
		self.variation_sales_firstyear_secondyear_variable = variation_sales_firstyear_secondyear_variable
		self.similarity = similarity

		self.fitted = False
		self.predicted = False

		if stock_ids is None:
			stock_ids = self.data[self.data[f"{self.quantity_stock_variable_prefix}_{self.second_year}"].notna()][id_variable].unique()

		# Get only the rows that are in the stock_ids
		self.stock_data = self.data[self.data[self.id_variable].isin(stock_ids)].reset_index(drop=True)

		print("Calculating impairment index...")
		# Calculate impairment index
		self.impairment_indexs = self.calculate_impairment_index_formula(data=self.stock_data)

		print("Calculating monthly data...")
		# Get the monthly data
		self.data_monthly = self.get_monthly_data(data=self.stock_data)

		print("Creating auto arima model...")
		# Create auto arima model and forecast for each stock
		self.auto_arima_indexs = self.create_auto_arima_and_forecast(data=self.data_monthly)

		print("Creating auto encoder model...")
		# Create auto encoder model
		self.auto_encoder_indexs = self.create_auto_encoder(data=self.stock_data, auto_arima_indicators=self.auto_arima_indexs)

		self.fitted = True
		print("Model fitted.")

	def predict(self, 
			tolerance: float = 1.5, 
			indexs_weights: dict = {'impairment': 0.1, 'auto_arima': 0.1, 'auto_encoder': 1}, 
			threshold: bool = True
			) -> pd.Series:
		"""
		Predict the fair price for the stock based on the fitted model.

		Parameters
		-----------
		tolerance: float
			The tolerance to use when calculating the fair price. More tolerance means tendence to less depreciation in the stock.

		indexs_weights: dict
			The weights to apply to the different indexs when merging them. The keys must be 'auto_arima', 'auto_encoder' and 'impairment'.

		Returns
		--------
		fair_price: pd.Series
			The fair price predicted for the stock in the second year.
		"""
		assert tolerance > 0, "Tolerance must be greater than 0."
		assert self.fitted, "Model must be fitted before predicting."
		assert set(indexs_weights.keys()) == {'auto_arima', 'auto_encoder', 'impairment'}, "indexs_weights must have the keys 'auto_arima', 'auto_encoder' and 'impairment'."
		assert threshold in ['mode', 'mean'] if isinstance(threshold, str) else (isinstance(threshold, (int, float)) and threshold > 0), "Threshold must be 'mode', 'mean' or a positive number."

		self.tolerance = tolerance
		self.indexs_weights = indexs_weights
		self.threshold = threshold

		# Interpret the indexes
		self.data_indexs_interpreted = self.indexs_interpretation(data=self.stock_data, impairment_index=self.impairment_indexs, auto_encoder_indexs=self.auto_encoder_indexs, auto_arima_indexs=self.auto_arima_indexs)
		
		self.predicted = True

		if 'index' in self.data_indexs_interpreted.columns:
			self.data_indexs_interpreted.drop(columns=['index'], inplace=True)

		self.forecasts_to_excel("./data/results_inventory_impairment.xlsx")

		return self.data_indexs_interpreted
	
	def explain(self, scale=True):
		"""
		Explain the model using an Explainable Boosting Regressor (EBM). This will show the most important features for the model.

		Parameters
		-----------
		scale: bool
			Wether to scale the data or not

		Returns
		--------
		ebm: ExplainableBoostingRegressor
			The EBM model used to explain the model.

		X: pd.DataFrame
			The data used to explain the model.
		
		y: pd.Series
			The target variable used to explain the model.
		"""
		assert self.fitted and self.predicted, "Model must be fitted and predicted to be explained."

		# Prepare data
		data = self.data_indexs_interpreted.copy()
		
		X = data[
			[
				f"{self.number_of_units_sold_variable_prefix}_{self.first_year}",
				f"{self.number_of_units_sold_variable_prefix}_{self.second_year}",
				f"{self.sales_variable_prefix}_{self.first_year}",
				f"{self.sales_variable_prefix}_{self.second_year}",
				f"{self.unitary_sale_price_variable_prefix}_{self.first_year}",
				f"{self.unitary_sale_price_variable_prefix}_{self.second_year}",
				self.variation_units_firstyear_secondyear_variable,
				self.proportion_variation_units_firstyear_secondyear_variable,
				self.variation_sales_firstyear_secondyear_variable,
				self.proportion_variation_sales_firstyear_secondyear_variable,
				self.variation_unitary_sale_price_firstyear_secondyear_variable,
				self.proportion_variation_unitary_sale_price_firstyear_secondyear_variable,
				f"{self.unitary_cost_stock_variable_prefix}_{self.second_year}",
				f"{self.quantity_stock_variable_prefix}_{self.second_year}",
				f"{self.total_value_stock_variable_prefix}_{self.second_year}",
				f"{self.last_exit_days_variable}",
				f"{self.last_entry_days_variable}",
			]
		]
		
		y = data['fair_price'] / data[f"{self.unitary_cost_stock_variable_prefix}_{self.second_year}"]

		# Fill NaN values
		y = y.fillna(0)

		# Scale the data
		if scale:
			scaler = MinMaxScaler()
			X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
		

		# Split data into train and test sets
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

		# Create the EBM model
		ebm = ExplainableBoostingRegressor(random_state=self.random_state)
		ebm.fit(X_train, y_train)

		# Predict the test data
		y_pred = ebm.predict(X_test)

		# Calculate the mean squared error
		mse = mean_squared_error(y_test, y_pred)
		print(f"Mean Squared Error for the EBM model used to explain the model: {mse}")

		# Show the explanation
		ebm_global = ebm.explain_global()
		show(ebm_global)
		
		return ebm, X, y

