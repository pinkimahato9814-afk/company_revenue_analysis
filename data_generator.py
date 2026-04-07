import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import config

class DataGenerator:
    def __init__(self):
        self.num_sales = config.NUM_SALES
        self.num_products = config.NUM_PRODUCTS
        self.num_customers = config.NUM_CUSTOMERS
        self.num_stores = config.NUM_STORES
        self.num_employees = config.NUM_EMPLOYEES
        
        self.categories = ['Electronics', 'Home & Garden', 'Apparel', 'Grocery', 'Health & Beauty']
        self.regions = ['North', 'South', 'East', 'West', 'Central']
        self.payment_methods = ['Credit Card', 'Cash', 'Digital Wallet', 'Debit Card']
        
    def generate_products(self):
        product_ids = [f"PRD{i:04d}" for i in range(1, self.num_products + 1)]
        data = {
            'product_id': product_ids,
            'name': [f"Product {i}" for i in range(1, self.num_products + 1)],
            'category': [random.choice(self.categories) for _ in range(self.num_products)],
            'cost': [round(random.uniform(5, 500), 2) for _ in range(self.num_products)],
        }
        # Add price with a margin (20% - 100%)
        df = pd.DataFrame(data)
        df['price'] = df['cost'] * (1 + np.random.uniform(0.2, 1.0, size=self.num_products))
        df['price'] = df['price'].round(2)
        
        # Introduce some missing costs
        idx = random.sample(range(self.num_products), 5)
        df.loc[idx, 'cost'] = np.nan
        
        df.to_csv(config.PRODUCTS_FILE, index=False)
        return df

    def generate_customers(self):
        customer_ids = [f"CUST{i:04d}" for i in range(1, self.num_customers + 1)]
        data = {
            'customer_id': customer_ids,
            'name': [f"Customer {i}" for i in range(1, self.num_customers + 1)],
            'city': [f"City {random.randint(1, 20)}" for _ in range(self.num_customers)],
            'registration_date': [self._random_date(config.START_DATE, config.END_DATE) for _ in range(self.num_customers)]
        }
        df = pd.DataFrame(data)
        df.to_csv(config.CUSTOMERS_FILE, index=False)
        return df

    def generate_stores(self):
        store_ids = [f"ST{i:03d}" for i in range(1, self.num_stores + 1)]
        data = {
            'store_id': store_ids,
            'city': [f"Store City {i}" for i in range(1, self.num_stores + 1)],
            'region': [random.choice(self.regions) for _ in range(self.num_stores)],
            'manager': [f"Manager {i}" for i in range(1, self.num_stores + 1)]
        }
        df = pd.DataFrame(data)
        df.to_csv(config.STORES_FILE, index=False)
        return df

    def generate_employees(self):
        employee_ids = [f"EMP{i:03d}" for i in range(1, self.num_employees + 1)]
        data = {
            'employee_id': employee_ids,
            'store_id': [f"ST{random.randint(1, self.num_stores):03d}" for _ in range(self.num_employees)],
            'name': [f"Employee {i}" for i in range(1, self.num_employees + 1)],
            'role': [random.choice(['Sales Rep', 'Store Manager', 'Cashier']) for _ in range(self.num_employees)]
        }
        df = pd.DataFrame(data)
        df.to_csv(config.EMPLOYEES_FILE, index=False)
        return df

    def generate_sales(self, products_df, customers_df, stores_df, employees_df):
        sales_data = []
        start_dt = datetime.strptime(config.START_DATE, "%Y-%m-%d")
        end_dt = datetime.strptime(config.END_DATE, "%Y-%m-%d")
        delta_days = (end_dt - start_dt).days

        for i in range(1, self.num_sales + 1):
            # Weight dates for seasonality (Dec, weekends)
            day_offset = random.randint(0, delta_days)
            dt = start_dt + timedelta(days=day_offset)
            
            # Boost sales in December
            if dt.month == 12:
                if random.random() < 0.3: # extra 30% chance for another sale
                     day_offset = random.randint(0, delta_days) # just dummy logic to spread
            
            customer = customers_df.sample(1).iloc[0]
            product = products_df.sample(1).iloc[0]
            store = stores_df.sample(1).iloc[0]
            # Employee must be from the same store
            store_employees = employees_df[employees_df['store_id'] == store['store_id']]
            if store_employees.empty:
                employee_id = "EMP000" # Fallback
            else:
                employee_id = store_employees.sample(1).iloc[0]['employee_id']

            quantity = random.randint(1, 5)
            # Add outliers in quantity
            if random.random() < 0.01:
                quantity = random.randint(20, 50)

            discount_pct = 0
            if random.random() < 0.3: # 30% sales have discount
                discount_pct = random.choice([5, 10, 15, 20, 25])
            
            is_returned = random.random() < 0.05 # 5% return rate

            sales_data.append({
                'transaction_id': f"TXN{i:06d}",
                'date': dt.strftime("%Y-%m-%d"),
                'hour': random.randint(8, 21), # Store hours 8 AM to 10 PM
                'customer_id': customer['customer_id'],
                'product_id': product['product_id'],
                'store_id': store['store_id'],
                'employee_id': employee_id,
                'quantity': quantity,
                'discount_pct': discount_pct,
                'payment_method': random.choice(self.payment_methods),
                'is_returned': is_returned
            })

        df = pd.DataFrame(sales_data)
        # Introduce some missing dates
        idx = random.sample(range(self.num_sales), 10)
        df.loc[idx, 'date'] = np.nan

        df.to_csv(config.SALES_FILE, index=False)
        return df

    def _random_date(self, start, end):
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        delta = end_dt - start_dt
        int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
        random_second = random.randrange(int_delta)
        return (start_dt + timedelta(seconds=random_second)).strftime("%Y-%m-%d")

    def generate_all(self):
        print("Generating data...")
        products = self.generate_products()
        customers = self.generate_customers()
        stores = self.generate_stores()
        employees = self.generate_employees()
        self.generate_sales(products, customers, stores, employees)
        print("Data generation complete.")

if __name__ == "__main__":
    dg = DataGenerator()
    dg.generate_all()
