import pandas as pd
import numpy as np
# DATA CLEAN
info_df = pd.read_excel('data/inventory_data_semi_raw.xlsx', sheet_name='Inf.')
accounting_info_1_df = pd.read_excel('data/inventory_data_semi_raw.xlsx', sheet_name='info_comptable_1')
accounting_info_2_df = pd.read_excel('data/inventory_data_semi_raw.xlsx', sheet_name='info_comptable_2')
sales_2023_df = pd.read_excel('data/inventory_data_semi_raw.xlsx', sheet_name='Vendes 2023')
sales_2022_df = pd.read_excel('data/inventory_data_semi_raw.xlsx', sheet_name='Vendes 2022')
catalog_prices_2023_df = pd.read_excel('data/inventory_data_semi_raw.xlsx', sheet_name='Preus de venda catàleg 2023')
last_exit_entry_df = pd.read_excel('data/inventory_data_semi_raw.xlsx', sheet_name='Darrera entrada-sortida')
stock_31_12_2023_df = pd.read_excel('data/inventory_data_semi_raw.xlsx', sheet_name='Stock 31.12.23')

# Strip all column names
for col in info_df.columns:
	info_df.rename(columns={col: col.strip()}, inplace=True)

for col in accounting_info_1_df.columns:
	accounting_info_1_df.rename(columns={col: col.strip()}, inplace=True)

for col in accounting_info_2_df.columns:
	accounting_info_2_df.rename(columns={col: col.strip()}, inplace=True)

for col in sales_2023_df.columns:
	sales_2023_df.rename(columns={col: col.strip()}, inplace=True)

for col in sales_2022_df.columns:
	sales_2022_df.rename(columns={col: col.strip()}, inplace=True)

for col in catalog_prices_2023_df.columns:
	catalog_prices_2023_df.rename(columns={col: col.strip()}, inplace=True)

for col in last_exit_entry_df.columns:
	last_exit_entry_df.rename(columns={col: col.strip()}, inplace=True)

for col in stock_31_12_2023_df.columns:
	stock_31_12_2023_df.rename(columns={col: col.strip()}, inplace=True)

# Drop 'Mon.' column
accounting_info_1_df.drop(columns=['Mon.', 'Mon..1'], inplace=True)
accounting_info_2_df.drop(columns=['Mon.', 'Mon..1'], inplace=True)
sales_2023_df.drop(columns=['Mon.', 'UMB'], inplace=True)
sales_2022_df.drop(columns=['Mon.', 'UMB'], inplace=True)
catalog_prices_2023_df.drop(columns=['Mon.'], inplace=True)
stock_31_12_2023_df.drop(columns=['Mon.', 'Mon..1', 'UMB'], inplace=True)

# Rename columns
accounting_info_1_df.rename(columns={'Núm compte': 'num_compte', 'Etiqueta': 'etiqueta', 'Final 2023': 'final_2023', 'Final 2022': 'final_2022'}, inplace=True)
accounting_info_2_df.rename(columns={'Núm compte': 'num_compte', 'Etiqueta': 'etiqueta', 'Exercici 2023': 'exercici_2023', 'Exercici 2022': 'exercici_2022'}, inplace=True)
sales_2023_df.rename(columns={'Material (Referencia)': 'material', 'Quantitat facturada': 'quantitat', 'Ventes (sense IVA)': 'vendes'}, inplace=True)
sales_2022_df.rename(columns={'Material (Referencia)': 'material', 'Quantitat facturada': 'quantitat', 'Ventes (sense IVA)': 'vendes'}, inplace=True)
catalog_prices_2023_df.rename(columns={'Material (Referencia)': 'material', 'P.venda unit': 'preu_unitat'}, inplace=True)
last_exit_entry_df.rename(columns={'Material (Referencia)': 'material', 'Data darrera sortida': 'data_sortida', 'Data darrera entrada': 'data_entrada'}, inplace=True)
stock_31_12_2023_df.rename(columns={'Material (Referencia)': 'material', 'Stock total': 'stock', 'Valor total': 'valor_total', 'Cost unit': 'cost_unitat'}, inplace=True)

# Save it to a new excel file
with pd.ExcelWriter('data/inventory_data_clean.xlsx') as writer:
	accounting_info_1_df.to_excel(writer, sheet_name='info_comptable_1', index=False)
	accounting_info_2_df.to_excel(writer, sheet_name='info_comptable_2', index=False)
	sales_2023_df.to_excel(writer, sheet_name='vendes_2023', index=False)
	sales_2022_df.to_excel(writer, sheet_name='vendes_2022', index=False)
	catalog_prices_2023_df.to_excel(writer, sheet_name='preus_venda_cataleg_2023', index=False)
	last_exit_entry_df.to_excel(writer, sheet_name='darrera_entrada_sortida', index=False)
	stock_31_12_2023_df.to_excel(writer, sheet_name='stock_final_2023', index=False)
	
# DATA MERGED

# Load the datasets
vendes_2022_df = pd.read_excel('./data/inventory_data_clean.xlsx', sheet_name='vendes_2022')
vendes_2023_df = pd.read_excel('./data/inventory_data_clean.xlsx', sheet_name='vendes_2023')
preus_venda_cataleg_2023_df = pd.read_excel('./data/inventory_data_clean.xlsx', sheet_name='preus_venda_cataleg_2023')
darrera_entrada_sortida_df = pd.read_excel('./data/inventory_data_clean.xlsx', sheet_name='darrera_entrada_sortida')
stock_final_2023_df = pd.read_excel('./data/inventory_data_clean.xlsx', sheet_name='stock_final_2023')

# Rename duplicated columns
vendes_2022_df.rename(columns={'quantitat': 'quantitat_2022', 'vendes': 'vendes_2022'}, inplace=True)
vendes_2023_df.rename(columns={'quantitat': 'quantitat_2023', 'vendes': 'vendes_2023'}, inplace=True)
preus_venda_cataleg_2023_df.rename(columns={'preu_unitat': 'preu_unitat_2023'}, inplace=True)
stock_final_2023_df.rename(columns={'stock': 'stock_final_2023', 'valor_total': 'valor_total_2023', 'cost_unitat': 'cost_unitat_2023'}, inplace=True)
darrera_entrada_sortida_df.rename(columns={'data_entrada': 'darrera_data_entrada', 'data_sortida': 'darrera_data_sortida'}, inplace=True)

# Merge the datasets
merged_df = pd.merge(vendes_2022_df, vendes_2023_df, on='material', how='outer')
merged_df = pd.merge(merged_df, preus_venda_cataleg_2023_df, on='material', how='outer')
merged_df = pd.merge(merged_df, darrera_entrada_sortida_df, on='material', how='outer')
merged_df = pd.merge(merged_df, stock_final_2023_df, on='material', how='outer')

# Save the merged dataset
merged_df.to_excel('./data/inventory_data_merged.xlsx', index=False)

# DATA NEW
# Read the dataset
inventory_data_merged_df = pd.read_excel('./data/inventory_data_merged.xlsx')

inventory_data_merged_df.rename(columns={'quantitat_2022': 'unitats_2022', 'quantitat_2023': 'unitats_2023', 'valor_total_2023': 'valor_total_stock_2023', 'cost_unitat_2023': 'cost_unitari_stock_2023', 'preu_unitat_2023': 'preu_venda_unitari_2023', 'darrera_data_entrada': 'data_darrera_entrada', 'darrera_data_sortida': 'data_darrera_sortida'}, inplace=True)

# Change dates to days since the date
inventory_data_merged_df['data_darrera_entrada'] = pd.to_datetime(inventory_data_merged_df['data_darrera_entrada'], format='%m-%d-%Y')
inventory_data_merged_df['data_darrera_sortida'] = pd.to_datetime(inventory_data_merged_df['data_darrera_sortida'], format='%m-%d-%Y')
inventory_data_merged_df['dies_ultima_entrada'] = (pd.to_datetime('31-12-2023', format='%d-%m-%Y') - inventory_data_merged_df['data_darrera_entrada']).dt.days
inventory_data_merged_df['dies_ultima_sortida'] = (pd.to_datetime('31-12-2023', format='%d-%m-%Y') - inventory_data_merged_df['data_darrera_sortida']).dt.days
inventory_data_merged_df['diferencia_entrada_sortida'] = inventory_data_merged_df['dies_ultima_entrada'] - inventory_data_merged_df['dies_ultima_sortida']

# Add new variables
inventory_data_merged_df['preu_venda_unitari_2022'] = inventory_data_merged_df['vendes_2022'] / inventory_data_merged_df['unitats_2022']
# inventory_data_merged_df['preu_venda_unitari_2023'] = inventory_data_merged_df['vendes_2023'] / inventory_data_merged_df['unitats_2023'] # No és necessari, ja que ja tenim el preu de venda unitari del 2023

inventory_data_merged_df['variacio_preu_venda_unitari_2022_2023'] = inventory_data_merged_df['preu_venda_unitari_2023'] - inventory_data_merged_df['preu_venda_unitari_2022']
inventory_data_merged_df['proporcio_variacio_preu_venda_unitari_2022_2023'] = inventory_data_merged_df['variacio_preu_venda_unitari_2022_2023'] / inventory_data_merged_df['preu_venda_unitari_2022']

inventory_data_merged_df['variacio_unitats_2022_2023'] = inventory_data_merged_df['unitats_2023'] - inventory_data_merged_df['unitats_2022']
inventory_data_merged_df['proporcio_variacio_unitats_2022_2023'] = np.where(
    inventory_data_merged_df['unitats_2022'] != 0,
    inventory_data_merged_df['variacio_unitats_2022_2023'] / inventory_data_merged_df['unitats_2022'],
    100 
)
inventory_data_merged_df['proporcio_variacio_unitats_2022_2023'] = np.where(
    (inventory_data_merged_df['unitats_2023'] == 0) & (inventory_data_merged_df['unitats_2022'] == 0),
    0,
    inventory_data_merged_df['proporcio_variacio_unitats_2022_2023']
)

inventory_data_merged_df['variacio_vendes_2022_2023'] = inventory_data_merged_df['vendes_2023'] - inventory_data_merged_df['vendes_2022']
inventory_data_merged_df['proporcio_variacio_vendes_2022_2023'] = np.where(
    inventory_data_merged_df['vendes_2022'] != 0,
    inventory_data_merged_df['variacio_vendes_2022_2023'] / inventory_data_merged_df['vendes_2022'],
    100 
)

inventory_data_merged_df['proporcio_variacio_vendes_2022_2023'] = np.where(
    (inventory_data_merged_df['vendes_2023'] == 0) & (inventory_data_merged_df['vendes_2022'] == 0),
    0,
    inventory_data_merged_df['proporcio_variacio_vendes_2022_2023']
)

# Reorder columns
# material, unitats_2022, vendes_2022, preu_venda_unitari_2022, unitats_2023, vendes_2023, preu_venda_unitari_2023, 
# variacio_preu_venda_unitari_2022_2023, proporcio_variacio_preu_venda_unitari_2022_2023, 
# darrera_data_entrada, darrera_data_sortida, dies_ultima_entrada, dies_ultima_sortida, diferencia_entrada_sortida, 
# stock_final_2023, valor_total_stock_2023, cost_unitari_stock_2023

columns_before = inventory_data_merged_df.columns
inventory_data_merged_df = inventory_data_merged_df[['material', 'unitats_2022', 'vendes_2022', 'preu_venda_unitari_2022', 'unitats_2023', 'vendes_2023', 'preu_venda_unitari_2023', 'variacio_unitats_2022_2023', 'proporcio_variacio_unitats_2022_2023', 'variacio_vendes_2022_2023', 'proporcio_variacio_vendes_2022_2023', 'variacio_preu_venda_unitari_2022_2023', 'proporcio_variacio_preu_venda_unitari_2022_2023', 'data_darrera_entrada', 'dies_ultima_entrada', 'data_darrera_sortida', 'dies_ultima_sortida', 'diferencia_entrada_sortida', 'stock_final_2023', 'valor_total_stock_2023', 'cost_unitari_stock_2023']]
columns_after = inventory_data_merged_df.columns

# Delete the row with the material 'Total'
inventory_data_merged_df = inventory_data_merged_df[inventory_data_merged_df['material'] != 'TOTAL']

# Save the new dataset
inventory_data_merged_df.to_excel('./data/inventory_data_new.xlsx', index=False)

