import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from opendp_laplace_runner import run_opendp_laplace_experiments
from google_dp_laplace_runner import run_google_dp_laplace_experiments

# synthetic data generation
def generate_synthetic_data_master(num_rows: int, filepath: str, num_numerical_cols: int = 5, numerical_range: tuple[float, float] = (-100.0, 100.0), random_state: int = 42):
    if os.path.exists(filepath):
        return
    np.random.seed(random_state)
    data = {}
    for i in range(num_numerical_cols):
        data[f'value_col_{i+1}'] = np.random.uniform(numerical_range[0], numerical_range[1], num_rows).astype(float)
    df_gen = pd.DataFrame(data)
    df_gen.to_csv(filepath, index=False)
    print(f"'{filepath}' created with {num_rows} rows and {num_numerical_cols} data columns.")

num_data_columns_for_generation = 5
datasets_to_test_config = [
    {'name': 'Small (10k rows)', 'file': 'small_dataset.csv', 'rows': 10000},
    {'name': 'Medium (100k rows)', 'file': 'medium_dataset.csv', 'rows': 100000},
    {'name': 'Large (1M rows)', 'file': 'large_dataset.csv', 'rows': 1000000},
]

# generate datasets if they don't exist
print("checking if datasets exist")
for ds_info in datasets_to_test_config:
    generate_synthetic_data_master(ds_info['rows'], ds_info['file'], num_numerical_cols=num_data_columns_for_generation)


EPSILON_VALUES_CONFIG = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
NUM_RUNS_CONFIG = 30
SUM_LOWER_BOUND_CONFIG = -100.0
SUM_UPPER_BOUND_CONFIG = 100.0
SUM_COLUMN_CONFIG = 'value_col_1'

print("\nRunning OpenDP Laplace Experiments")
results_opendp = run_opendp_laplace_experiments(
    datasets_to_test_config,
    epsilon_values=EPSILON_VALUES_CONFIG,
    num_runs=NUM_RUNS_CONFIG,
    sum_lower_bound=SUM_LOWER_BOUND_CONFIG,
    sum_upper_bound=SUM_UPPER_BOUND_CONFIG,
    sum_column=SUM_COLUMN_CONFIG
)

print("\nRunning Google DP Laplace Experiments")
results_google_dp = run_google_dp_laplace_experiments(
    datasets_to_test_config,
    epsilon_values=EPSILON_VALUES_CONFIG,
    num_runs=NUM_RUNS_CONFIG,
    sum_lower_bound=SUM_LOWER_BOUND_CONFIG,
    sum_upper_bound=SUM_UPPER_BOUND_CONFIG,
    sum_column=SUM_COLUMN_CONFIG
)

# combine results for plotting
plot_data_list_combined = []
dataset_size_map = {ds['name']: ds['rows'] for ds in datasets_to_test_config}

# process OpenDP
for dataset_name, results_data in results_opendp.items():
    for query_type_key, query_results in results_data.items():
        query_base_name = 'Count' if 'count' in query_type_key else 'Sum'
        if query_results:
            for res in query_results:
                if res['mae'] != 'Error':
                    plot_data_list_combined.append({
                        'library': 'OpenDP',
                        'dataset_name': dataset_name,
                        'dataset_size': dataset_size_map[dataset_name],
                        'query': query_base_name,
                        'mechanism': 'Laplace',
                        'epsilon': res['epsilon'],
                        'mae': float(res['mae']),
                        'avg_time_ms': float(res['avg_time_ms'])
                    })

# process Google DP
if results_google_dp:
    for dataset_name, results_data in results_google_dp.items():
        for query_type_key, query_results in results_data.items():
            query_base_name = 'Count' if 'count' in query_type_key else 'Sum'
            if query_results:
                for res in query_results:
                    if res['mae'] != 'Error':
                        plot_data_list_combined.append({
                            'library': 'GoogleDP',
                            'dataset_name': dataset_name,
                            'dataset_size': dataset_size_map[dataset_name],
                            'query': query_base_name,
                            'mechanism': 'Laplace',
                            'epsilon': res['epsilon'],
                            'mae': float(res['mae']),
                            'avg_time_ms': float(res['avg_time_ms'])
                        })

if not plot_data_list_combined:
    print("No data to plot. Exiting.")
    exit()

plot_df_combined = pd.DataFrame(plot_data_list_combined)

if not plot_df_combined.empty:
    results_filepath_csv = "combined_laplace_results.csv"
    plot_df_combined.to_csv(results_filepath_csv, index=False)
else:
    print("Combined DataFrame is empty")
