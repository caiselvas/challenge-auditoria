import numpy as np
import matplotlib.pyplot as plt

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

def generate_monthly_sales(yearly_total, monthly_avg_sales, last_month):
    # Generate monthly data
    monthly_sales = np.random.normal(monthly_avg_sales, monthly_avg_sales * 0.1, last_month)

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

def main(last_month_2022=6, last_month_2023=6):
    # Yearly totals
    vendes_2022 = 0
    vendes_2023 = 4800000000000

    # Monthly data
    months_in_year = 12
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Calculate monthly average values
    monthly_avg_sales_2022 = vendes_2022 / last_month_2022
    monthly_avg_sales_2023 = vendes_2023 / last_month_2023

    # Generate monthly sales data
    sales_2022 = generate_monthly_sales(vendes_2022, monthly_avg_sales_2022, last_month_2022)
    sales_2023 = generate_monthly_sales(vendes_2023, monthly_avg_sales_2023, last_month_2023)

    # Adjust monthly sales to match yearly total
    monthly_vendes_2022 = calculate_monthly_sales(vendes_2022, sales_2022, last_month_2022)
    monthly_vendes_2023 = calculate_monthly_sales(vendes_2023, sales_2023, last_month_2023)

    # Output
    print("Monthly sales for 2022:")
    print(monthly_vendes_2022)
    print("Monthly sales for 2023:")
    print(sum(monthly_vendes_2023), vendes_2023)
    print(sum(monthly_vendes_2022), vendes_2022)

    # Plot monthly sales
    plot_monthly_sales(months, monthly_vendes_2022, monthly_vendes_2023, last_month_2022, last_month_2023)

if __name__ == "__main__":
    main(last_month_2022=8, last_month_2023=7)
