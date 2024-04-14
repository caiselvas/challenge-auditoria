import pandas as pd

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