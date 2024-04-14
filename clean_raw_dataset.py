import pandas as pd

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
