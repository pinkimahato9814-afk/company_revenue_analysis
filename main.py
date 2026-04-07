from data_generator import DataGenerator
from data_loader import DataLoader
from analysis import RevenueAnalysis
from visualization import Visualizer
from ml_model import MLModule
import os
import config

def main():
    print("=== Retail Store Revenue Analysis Project ===")
    
    # 1. Generate Data (if not exists or force regen)
    if not os.path.exists(config.SALES_FILE):
        dg = DataGenerator()
        dg.generate_all()
    else:
        print("Data already exists. Skipping generation.")

    # 2. Load and Preprocess Data
    loader = DataLoader()
    loader.load_data()
    merged_data = loader.get_merged_data()
    
    # 3. Initialize Visualizer
    viz = Visualizer()
    
    # 4. Run Business Analysis (30 Questions)
    analysis = RevenueAnalysis(merged_data, viz)
    analysis.run_all_questions()
    
    # 5. Run Machine Learning Models
    ml = MLModule(merged_data, viz)
    ml.run_all_ml()
    
    print("\nProject execution completed successfully.")
    print(f"All datasets are in the '{config.DATA_DIR}' folder.")
    print(f"All visualizations are in the '{config.PLOTS_DIR}' folder.")

if __name__ == "__main__":
    main()
