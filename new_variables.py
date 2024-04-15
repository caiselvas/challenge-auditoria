import pandas as pd

# Read the dataset
inventory_data_merged_df = pd.read_excel('./data/inventory_data_merged.xlsx')


# Change dates to days since the date
inventory_data_merged_df['darrera_data_entrada'] = pd.to_datetime(inventory_data_merged_df['darrera_data_entrada'], format='%m-%d-%Y')
inventory_data_merged_df['darrera_data_sortida'] = pd.to_datetime(inventory_data_merged_df['darrera_data_sortida'], format='%m-%d-%Y')
inventory_data_merged_df['dies_ultima_entrada'] = (pd.to_datetime('31-12-2023', format='%d-%m-%Y') - inventory_data_merged_df['darrera_data_entrada']).dt.days
inventory_data_merged_df['dies_ultima_sortida'] = (pd.to_datetime('31-12-2023', format='%d-%m-%Y') - inventory_data_merged_df['darrera_data_sortida']).dt.days
inventory_data_merged_df['diferencia_entrada_sortida'] = inventory_data_merged_df['dies_ultima_entrada'] - inventory_data_merged_df['dies_ultima_sortida']

# Save the new dataset
inventory_data_merged_df.to_excel('./data/inventory_data_new.xlsx', index=False)

