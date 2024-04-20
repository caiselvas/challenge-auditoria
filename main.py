from inventory_impairment_class import InventoryImpairment
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Load data
data = pd.read_excel('./data/inventory_data_new.xlsx')

model = InventoryImpairment()

# Fit the model
model.fit(data, variability=0.5)

# Predict the impairment
impairment = model.predict()

# Explain the model
model.explain()
