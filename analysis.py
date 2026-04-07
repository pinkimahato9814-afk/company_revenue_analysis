import pandas as pd
import numpy as np
from visualization import Visualizer

class RevenueAnalysis:
    def __init__(self, merged_data, visualizer):
        self.df = merged_data
        self.viz = visualizer
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['month_year'] = self.df['date'].dt.to_period('M').astype(str)
        self.df['day_name'] = self.df['date'].dt.day_name()

    def run_all_questions(self):
        print("\n--- Starting Business Analysis (30 Questions) ---\n")
        self.q1_total_revenue()
        self.q2_monthly_revenue_trend()
        self.q3_yearly_growth()
        self.q4_best_selling_products()
        self.q5_worst_selling_products()
        self.q6_highest_profit_categories()
        self.q7_low_margin_products()
        self.q8_top_customers()
        self.q9_customer_segmentation_info()
        self.q10_store_performance()
        self.q11_payment_method_revenue()
        self.q12_discount_revenue_impact()
        self.q13_discount_profit_impact()
        self.q14_sales_by_weekday()
        self.q15_sales_by_month_seasonality()
        self.q16_sales_by_season()
        self.q17_return_rate_impact()
        self.q18_sales_volume_distribution()
        self.q19_salesperson_performance()
        self.q20_regional_performance()
        self.q21_average_order_value()
        self.q22_repeat_vs_new_customers()
        self.q23_revenue_prediction_info()
        self.q24_demand_forecasting_info()
        self.q25_anomaly_detection_info()
        self.q26_price_quantity_correlation()
        self.q27_revenue_by_hour()
        self.q28_cost_vs_profit()
        self.q29_most_profitable_stores()
        self.q30_customer_lifetime_value()

    def q1_total_revenue(self):
        total = self.df['net_revenue'].sum()
        print(f"Q1: Total Net Revenue: ${total:,.2f}")
        print("Importance: Overall health metric of the business.")
        # No chart for a single number, but we can show revenue vs cost
        data = pd.DataFrame({'Metric': ['Revenue', 'Cost', 'Profit'], 
                             'Value': [self.df['net_revenue'].sum(), self.df['total_cost'].sum(), self.df['net_profit'].sum()]})
        self.viz.plot_bar(data, 'Metric', 'Value', 'Overall Financial Overview', 'q1_financial_overview.png')

    def q2_monthly_revenue_trend(self):
        trend = self.df.groupby('month_year')['net_revenue'].sum().reset_index()
        print("Q2: Monthly Revenue Trend calculated.")
        self.viz.plot_line(trend, 'month_year', 'net_revenue', 'Monthly Revenue Trend', 'q2_monthly_trend.png', xlabel='Month-Year', ylabel='Net Revenue')

    def q3_yearly_growth(self):
        yearly = self.df.groupby('year')['net_revenue'].sum()
        growth = yearly.pct_change() * 100
        print(f"Q3: Yearly Revenue: \n{yearly}")
        self.viz.plot_bar(yearly.reset_index(), 'year', 'net_revenue', 'Yearly Revenue Comparison', 'q3_yearly_comparison.png')

    def q4_best_selling_products(self):
        top_10 = self.df.groupby('name')['net_revenue'].sum().sort_values(ascending=False).head(10).reset_index()
        print("Q4: Top 10 Best-Selling Products identified.")
        self.viz.plot_bar(top_10, 'name', 'net_revenue', 'Top 10 Products by Revenue', 'q4_top_products.png', horizontal=True)

    def q5_worst_selling_products(self):
        bottom_10 = self.df.groupby('name')['net_revenue'].sum().sort_values(ascending=True).head(10).reset_index()
        print("Q5: Bottom 10 Worst-Selling Products identified.")
        self.viz.plot_bar(bottom_10, 'name', 'net_revenue', 'Bottom 10 Products by Revenue', 'q5_bottom_products.png', horizontal=True)

    def q6_highest_profit_categories(self):
        cat_profit = self.df.groupby('category')['net_profit'].sum().sort_values(ascending=False).reset_index()
        print("Q6: Most Profitable Categories identified.")
        self.viz.plot_bar(cat_profit, 'category', 'net_profit', 'Profit by Category', 'q6_category_profit.png')

    def q7_low_margin_products(self):
        # Margin = Profit / Revenue
        prod_margin = self.df.groupby('name').agg({'net_profit':'sum', 'net_revenue':'sum'})
        prod_margin['margin'] = prod_margin['net_profit'] / prod_margin['net_revenue']
        low_margin = prod_margin.sort_values('margin').head(10).reset_index()
        print("Q7: Identified products with the lowest profit margins.")
        self.viz.plot_bar(low_margin, 'name', 'margin', 'Bottom 10 Product Margins', 'q7_low_margins.png', horizontal=True)

    def q8_top_customers(self):
        top_cust = self.df.groupby('name_cust')['net_revenue'].sum().sort_values(ascending=False).head(10).reset_index()
        print("Q8: Top 10 Customers by Revenue identified.")
        self.viz.plot_bar(top_cust, 'name_cust', 'net_revenue', 'Top 10 Customers', 'q8_top_customers.png', horizontal=True)

    def q9_customer_segmentation_info(self):
        print("Q9: Customer Segmentation - This is handled in the ML module using KMeans.")

    def q10_store_performance(self):
        store_perf = self.df.groupby('store_id')['net_revenue'].sum().sort_values(ascending=False).reset_index()
        print("Q10: Store performance calculated.")
        self.viz.plot_bar(store_perf, 'store_id', 'net_revenue', 'Revenue by Store', 'q10_store_performance.png')

    def q11_payment_method_revenue(self):
        pm_rev = self.df.groupby('payment_method')['net_revenue'].sum().reset_index()
        print("Q11: Revenue distribution by payment method.")
        self.viz.plot_pie(pm_rev['net_revenue'], pm_rev['payment_method'], 'Revenue by Payment Method', 'q11_payment_method.png')

    def q12_discount_revenue_impact(self):
        # Bin discounts
        self.df['discount_group'] = pd.cut(self.df['discount_pct'], bins=[-1, 0, 10, 20, 30], labels=['No Discount', 'Low (1-10%)', 'Medium (11-20%)', 'High (21-30%)'])
        disc_rev = self.df.groupby('discount_group', observed=True)['net_revenue'].mean().reset_index()
        print("Q12: Average revenue per transaction by discount group.")
        self.viz.plot_bar(disc_rev, 'discount_group', 'net_revenue', 'Avg Revenue by Discount Group', 'q12_discount_revenue.png')

    def q13_discount_profit_impact(self):
        disc_profit = self.df.groupby('discount_group', observed=True)['net_profit'].mean().reset_index()
        print("Q13: Average profit per transaction by discount group.")
        self.viz.plot_bar(disc_profit, 'discount_group', 'net_profit', 'Avg Profit by Discount Group', 'q13_discount_profit.png')

    def q14_sales_by_weekday(self):
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_sales = self.df.groupby('day_name')['net_revenue'].sum().reindex(order).reset_index()
        print("Q14: Revenue by day of the week.")
        self.viz.plot_bar(weekday_sales, 'day_name', 'net_revenue', 'Revenue by Weekday', 'q14_weekday_sales.png')

    def q15_sales_by_month_seasonality(self):
        month_sales = self.df.groupby('month')['net_revenue'].sum().reset_index()
        print("Q15: Aggregated monthly revenue for seasonality analysis.")
        self.viz.plot_bar(month_sales, 'month', 'net_revenue', 'Revenue by Month (Aggregated)', 'q15_monthly_seasonality.png')

    def q16_sales_by_season(self):
        def get_season(m):
            if m in [12, 1, 2]: return 'Winter'
            if m in [3, 4, 5]: return 'Spring'
            if m in [6, 7, 8]: return 'Summer'
            return 'Fall'
        self.df['season'] = self.df['month'].apply(get_season)
        season_sales = self.df.groupby('season')['net_revenue'].sum().reset_index()
        print("Q16: Revenue by season.")
        self.viz.plot_bar(season_sales, 'season', 'net_revenue', 'Revenue by Season', 'q16_seasonal_sales.png')

    def q17_return_rate_impact(self):
        return_summary = self.df.groupby('is_returned')['revenue'].count().reset_index()
        return_summary.columns = ['is_returned', 'transaction_count']
        print("Q17: Return rate impact - Number of returned transactions.")
        self.viz.plot_pie(return_summary['transaction_count'], ['Sold', 'Returned'], 'Return vs Sold Transactions', 'q17_return_impact.png')

    def q18_sales_volume_distribution(self):
        print("Q18: Distribution of transaction quantities.")
        self.viz.plot_histogram(self.df, 'quantity', 'Distribution of Order Quantities', 'q18_quantity_dist.png')

    def q19_salesperson_performance(self):
        emp_perf = self.df.groupby('employee_id')['net_revenue'].sum().sort_values(ascending=False).head(10).reset_index()
        print("Q19: Top 10 Employees by Revenue.")
        self.viz.plot_bar(emp_perf, 'employee_id', 'net_revenue', 'Top 10 Employees Performance', 'q19_employee_perf.png')

    def q20_regional_performance(self):
        reg_perf = self.df.groupby('region')['net_revenue'].sum().reset_index()
        print("Q20: Regional revenue performance.")
        self.viz.plot_bar(reg_perf, 'region', 'net_revenue', 'Revenue by Region', 'q20_regional_perf.png')

    def q21_average_order_value(self):
        aov = self.df['net_revenue'].sum() / self.df['transaction_id'].nunique()
        print(f"Q21: Average Order Value (AOV): ${aov:,.2f}")

    def q22_repeat_vs_new_customers(self):
        cust_counts = self.df.groupby('customer_id')['transaction_id'].count()
        repeat = (cust_counts > 1).sum()
        new = (cust_counts == 1).sum()
        print(f"Q22: Repeat Customers: {repeat}, New Customers: {new}")
        self.viz.plot_pie([new, repeat], ['New', 'Repeat'], 'New vs Repeat Customers', 'q22_customer_type.png')

    def q23_revenue_prediction_info(self):
        print("Q23: Revenue Prediction - Handled in ML module using Random Forest.")

    def q24_demand_forecasting_info(self):
        print("Q24: Demand Forecasting - Handled in ML module.")

    def q25_anomaly_detection_info(self):
        print("Q25: Anomaly Detection - Handled in ML module using Isolation Forest.")

    def q26_price_quantity_correlation(self):
        print("Q26: Correlation between Price and Quantity.")
        self.viz.plot_scatter(self.df, 'price', 'quantity', 'Price vs Quantity Correlation', 'q26_price_quantity.png')

    def q27_revenue_by_hour(self):
        hour_rev = self.df.groupby('hour')['net_revenue'].sum().reset_index()
        print("Q27: Peak store hours by revenue.")
        self.viz.plot_line(hour_rev, 'hour', 'net_revenue', 'Revenue by Hour of Day', 'q27_hourly_revenue.png')

    def q28_cost_vs_profit(self):
        print("Q28: Product Cost vs Net Profit.")
        self.viz.plot_scatter(self.df, 'cost', 'net_profit', 'Product Cost vs Profit', 'q28_cost_profit.png', hue='category')

    def q29_most_profitable_stores(self):
        store_profit = self.df.groupby('store_id')['net_profit'].sum().sort_values(ascending=False).reset_index()
        print("Q29: Most profitable stores.")
        self.viz.plot_bar(store_profit, 'store_id', 'net_profit', 'Profit by Store', 'q29_store_profit.png')

    def q30_customer_lifetime_value(self):
        clv = self.df.groupby('name_cust')['net_revenue'].sum().sort_values(ascending=False).head(15).reset_index()
        print("Q30: Top 15 Customers by Lifetime Value (Total Revenue).")
        self.viz.plot_bar(clv, 'name_cust', 'net_revenue', 'Top 15 Customers by CLV', 'q30_clv.png', horizontal=True)
