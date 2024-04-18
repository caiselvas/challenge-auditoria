import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_monthly_sales(yearly_total, monthly_avg_sales, last_month):
    # Calculate surplus/deficit
    surplus_deficit = (yearly_total - np.sum(monthly_avg_sales)) / last_month
    monthly_sales = monthly_avg_sales + surplus_deficit
    while np.min(monthly_sales) <0:   
        # Adjust monthly data to match yearly total
        monthly_sales = monthly_avg_sales + surplus_deficit
        monthly_sales = np.maximum(monthly_sales, 0)
        surplus_deficit = yearly_total - np.sum(monthly_sales)

    return monthly_sales

def generate_monthly_sales(yearly_total, monthly_avg_sales, last_month, VARIABILITY):
    # Generate monthly data
    monthly_sales = np.random.normal(monthly_avg_sales, monthly_avg_sales * VARIABILITY, last_month)

    # Apply seasonal adjustments
    if last_month > 8:
        monthly_sales[0] *= 0.8
        monthly_sales[6] *= 1.2
        monthly_sales[6] *= 1.2
        monthly_sales[7] *= 1.3  # Increase sales in summer
        monthly_sales[8] *= 1.1

    return monthly_sales

def plot_monthly_sales(months, sales_2022, sales_2023, last_month_2022, last_month_2023):
    plt.figure(figsize=(10, 6))
    plt.plot(months[:last_month_2022], sales_2022, label='2022 Sales')
    plt.plot(months[:last_month_2023], sales_2023, label='2023 Sales')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.title('Monthly Sales Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(sales22 = 1000, sales23 = 1200, last_month_2022=6, last_month_2023=6, VARIABILITY=0.1):
    # Yearly totals
    vendes_2022 = sales22
    vendes_2023 = sales23

    # Monthly data
    months_in_year = 12
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Calculate monthly average values
    monthly_avg_sales_2022 = vendes_2022 / last_month_2022 if last_month_2022 != 0 else 0
    monthly_avg_sales_2023 = vendes_2023 / last_month_2023 if last_month_2023 != 0 else 0

    # Generate monthly sales data
    sales_2022 = generate_monthly_sales(vendes_2022, monthly_avg_sales_2022, last_month_2022, VARIABILITY)  if last_month_2022 != 0 else [0 for _ in range(12)]
    sales_2023 = generate_monthly_sales(vendes_2023, monthly_avg_sales_2023, last_month_2023, VARIABILITY)  if last_month_2023 != 0 else [0 for _ in range(12)]
    # Adjust monthly sales to match yearly total
    monthly_vendes_2022 = calculate_monthly_sales(vendes_2022, sales_2022, last_month_2022)  if last_month_2022 != 0 else [0 for _ in range(12)]
    monthly_vendes_2023 = calculate_monthly_sales(vendes_2023, sales_2023, last_month_2023)  if last_month_2023 != 0 else [0 for _ in range(12)]


    if last_month_2022 ==0:
        last_month_2022 = 12
    if last_month_2023 == 0:
        last_month_2023 = 12

    # Plot monthly sales
    plot_monthly_sales(months, monthly_vendes_2022, monthly_vendes_2023, last_month_2022, last_month_2023)
    return monthly_vendes_2022, monthly_vendes_2023

if __name__ == "__main__":
    data = pd.read_excel("C:/Users/Usuario/Documents/Projectes/ChallengeAuditoria/challenge-auditoria/data/inventory_data_stock.xlsx")

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for month_idx, month in enumerate(months):
        data[f'{month}_2022'] = 0
        data[f'{month}_2023'] = 0
    
    for index, row in data.iterrows():
        year = pd.to_datetime(row["data_darrera_sortida"]).year
        if year == 2024:
            month22 = 12
            month23 = 12
        elif year == 2023:
            month22 = 12
            month23 = pd.to_datetime(row["data_darrera_sortida"]).month
        elif year == 2022:
            month22 = pd.to_datetime(row["data_darrera_sortida"]).month
            month23 = 0
        elif year < 2022:
            month22 = 0
            month23 = 0
        else:
            month22 = 12
            month23 = 12
        ventes22 = row["vendes_2022"] if not pd.isna(row["vendes_2022"]) else 0
        ventes23 = row["vendes_2023"] if not pd.isna(row["vendes_2023"]) else 0
        values22, values23 = main(sales22 = ventes22, sales23 = ventes23 ,last_month_2022=month22, last_month_2023=month23, VARIABILITY=0)
        for month_idx, month in enumerate(months):
            row[f'{month}_2022'] = values22[month_idx] if month_idx < len(values22) else 0
            row[f'{month}_2023'] = values23[month_idx] if month_idx < len(values23) else 0
        print(pd.to_datetime(row["data_darrera_sortida"]).year)
        data.loc[index] = row
        print(data.loc[index])
    data.to_excel('data/inventory_data_month.xlsx')