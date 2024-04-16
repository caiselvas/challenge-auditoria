import pandas as pd

# Read the dataset
inventory_data_merged_df = pd.read_excel('./data/inventory_data_merged.xlsx')

inventory_data_merged_df.rename(columns={'quantitat_2022': 'unitats_2022', 'quantitat_2023': 'unitats_2023', 'valor_total_2023': 'valor_total_stock_2023', 'cost_unitat_2023': 'cost_unitari_stock_2023', 'preu_unitat_2023': 'preu_venda_unitari_2023'}, inplace=True)

# Change dates to days since the date
inventory_data_merged_df['darrera_data_entrada'] = pd.to_datetime(inventory_data_merged_df['darrera_data_entrada'], format='%m-%d-%Y')
inventory_data_merged_df['darrera_data_sortida'] = pd.to_datetime(inventory_data_merged_df['darrera_data_sortida'], format='%m-%d-%Y')
inventory_data_merged_df['dies_ultima_entrada'] = (pd.to_datetime('31-12-2023', format='%d-%m-%Y') - inventory_data_merged_df['darrera_data_entrada']).dt.days
inventory_data_merged_df['dies_ultima_sortida'] = (pd.to_datetime('31-12-2023', format='%d-%m-%Y') - inventory_data_merged_df['darrera_data_sortida']).dt.days
inventory_data_merged_df['diferencia_entrada_sortida'] = inventory_data_merged_df['dies_ultima_entrada'] - inventory_data_merged_df['dies_ultima_sortida']

# Add new variables
inventory_data_merged_df['preu_venda_unitari_2022'] = inventory_data_merged_df['vendes_2022'] / inventory_data_merged_df['unitats_2022']
# inventory_data_merged_df['preu_venda_unitari_2023'] = inventory_data_merged_df['vendes_2023'] / inventory_data_merged_df['unitats_2023'] # No és necessari, ja que ja tenim el preu de venda unitari del 2023
inventory_data_merged_df['variacio_preu_venda_unitari_2022_2023'] = inventory_data_merged_df['preu_venda_unitari_2023'] - inventory_data_merged_df['preu_venda_unitari_2022']
inventory_data_merged_df['proporcio_variacio_preu_venda_unitari_2022_2023'] = inventory_data_merged_df['variacio_preu_venda_unitari_2022_2023'] / inventory_data_merged_df['preu_venda_unitari_2022']

# Reorder columns
# material, unitats_2022, vendes_2022, preu_venda_unitari_2022, unitats_2023, vendes_2023, preu_venda_unitari_2023, 
# variacio_preu_venda_unitari_2022_2023, proporcio_variacio_preu_venda_unitari_2022_2023, 
# darrera_data_entrada, darrera_data_sortida, dies_ultima_entrada, dies_ultima_sortida, diferencia_entrada_sortida, 
# stock_final_2023, valor_total_stock_2023, cost_unitari_stock_2023

columns_before = inventory_data_merged_df.columns
inventory_data_merged_df = inventory_data_merged_df[['material', 'unitats_2022', 'vendes_2022', 'preu_venda_unitari_2022', 'unitats_2023', 'vendes_2023', 'preu_venda_unitari_2023', 'variacio_preu_venda_unitari_2022_2023', 'proporcio_variacio_preu_venda_unitari_2022_2023', 'darrera_data_entrada', 'dies_ultima_entrada', 'darrera_data_sortida', 'dies_ultima_sortida', 'diferencia_entrada_sortida', 'stock_final_2023', 'valor_total_stock_2023', 'cost_unitari_stock_2023']]
columns_after = inventory_data_merged_df.columns

# Save the new dataset
inventory_data_merged_df.to_excel('./data/inventory_data_new.xlsx', index=False)

