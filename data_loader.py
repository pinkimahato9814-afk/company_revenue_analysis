import pandas as pd
import config

class DataLoader:
    def __init__(self):
        self.sales = None
        self.products = None
        self.customers = None
        self.stores = None
        self.employees = None

    def load_data(self):
        print("Loading data from CSV files...")
        self.sales = pd.read_csv(config.SALES_FILE)
        self.products = pd.read_csv(config.PRODUCTS_FILE)
        self.customers = pd.read_csv(config.CUSTOMERS_FILE)
        self.stores = pd.read_csv(config.STORES_FILE)
        self.employees = pd.read_csv(config.EMPLOYEES_FILE)
        
        self._preprocess()
        return self.sales, self.products, self.customers, self.stores, self.employees

    def _preprocess(self):
        # Convert dates
        self.sales['date'] = pd.to_datetime(self.sales['date'])
        self.customers['registration_date'] = pd.to_datetime(self.customers['registration_date'])
        
        # Handle missing values in sales date (fill with median or drop - let's drop for simplicity in this project)
        self.sales.dropna(subset=['date'], inplace=True)
        
        # Handle missing values in product cost
        self.products['cost'] = self.products['cost'].fillna(self.products['cost'].median())
        
        # Merge datasets for analysis
        self.merged_data = self.sales.merge(self.products, on='product_id')
        self.merged_data = self.merged_data.merge(self.customers, on='customer_id', suffixes=('', '_cust'))
        self.merged_data = self.merged_data.merge(self.stores, on='store_id', suffixes=('', '_store'))
        
        # Calculate derived columns
        # Revenue = Price * Quantity * (1 - Discount)
        self.merged_data['revenue'] = self.merged_data['price'] * self.merged_data['quantity'] * (1 - self.merged_data['discount_pct'] / 100)
        
        # Total Cost = Cost * Quantity
        self.merged_data['total_cost'] = self.merged_data['cost'] * self.merged_data['quantity']
        
        # Profit = Revenue - Total Cost
        self.merged_data['profit'] = self.merged_data['revenue'] - self.merged_data['total_cost']
        
        # Handle returns: if returned, revenue and profit should be adjusted (set to 0 for revenue/profit impact)
        # Or better, keep them and use is_returned for analysis. Let's create 'net_revenue'
        self.merged_data['net_revenue'] = self.merged_data.apply(lambda x: 0 if x['is_returned'] else x['revenue'], axis=1)
        self.merged_data['net_profit'] = self.merged_data.apply(lambda x: -x['total_cost'] if x['is_returned'] else x['profit'], axis=1)

    def get_merged_data(self):
        return self.merged_data
