import pandas as pd
import numpy as np
import os
import time

try:
    import pydp as dp
    from pydp.algorithms.laplacian import BoundedSum as GoogleLaplacianBoundedSum
    from pydp.algorithms.laplacian import Count as GoogleLaplacianCount
except ImportError:
    print("Error")
    dp = None

# configuration
DEFAULT_EPSILON_VALUES = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
DEFAULT_NUM_RUNS = 100
DEFAULT_SUM_LOWER_BOUND = -100.0
DEFAULT_SUM_UPPER_BOUND = 100.0
DEFAULT_SUM_COLUMN = 'value_col_1'

class GoogleDPCountLaplace:
    def __init__(self, epsilon):
        self.epsilon = epsilon
    def run(self, data_list_len):
        count_obj = GoogleLaplacianCount(epsilon=self.epsilon)
        if data_list_len > 0:
             count_obj.add_entries([1] * data_list_len)
        return count_obj.result()

class GoogleDPBoundedSumLaplace:
    def __init__(self, epsilon, lower_bound, upper_bound):
        self.epsilon = epsilon
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
    def run(self, data_list):
        bs = GoogleLaplacianBoundedSum(
            epsilon=self.epsilon,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            dtype='float'
        )
        bs.add_entries(data_list)
        return bs.result()

def run_google_dp_laplace_experiments(datasets_to_test,
                                      epsilon_values=DEFAULT_EPSILON_VALUES,
                                      num_runs=DEFAULT_NUM_RUNS,
                                      sum_lower_bound=DEFAULT_SUM_LOWER_BOUND,
                                      sum_upper_bound=DEFAULT_SUM_UPPER_BOUND,
                                      sum_column=DEFAULT_SUM_COLUMN):
    if dp is None:
        print("Google DP library not available.")
        return {}
    all_results_summary = {}

    for ds_info in datasets_to_test:
        dataset_name = ds_info['name']
        dataset_file = ds_info['file']
        print(f"\n\n Processing Dataset (Google DP): {dataset_name} ({dataset_file})")
        try:
            df = pd.read_csv(dataset_file)
            for col in df.columns:
                if df[col].dtype != float and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].astype(float)
        except Exception as e:
            print(f"Error loading dataset {dataset_file}: {e}")
            continue

        true_count = len(df)
        if sum_column not in df.columns or df[sum_column].dtype != float:
            print(f"Sum column issue in {dataset_file}.")
            continue
        true_sum = df[sum_column].sum()
        print(f"True Count: {true_count}, True Sum: {true_sum:.4f}")

        current_dataset_results = {'count_laplace': [], 'sum_laplace': []}
        
        # count
        data_for_count_list_len = len(df)
        print(f"\nTesting Count Query for {dataset_name} (Google DP):")
        for eps in epsilon_values:
            noisy_results = []
            start_time = time.time()
            try:
                google_count_runner = GoogleDPCountLaplace(epsilon=eps)
                for _ in range(num_runs):
                    noisy_results.append(google_count_runner.run(data_for_count_list_len))
            except Exception as e:
                print(f"  Epsilon {eps}: Error - {e}")
                current_dataset_results['count_laplace'].append({'epsilon': eps, 'avg_noisy_result': 'Error', 'mae': 'Error', 'avg_time_ms': 'Error'})
                continue
            avg_time = (time.time() - start_time) / num_runs
            avg_noisy = np.mean(noisy_results)
            mae = np.mean(np.abs(np.array(noisy_results) - true_count))
            current_dataset_results['count_laplace'].append({'epsilon': eps, 'avg_noisy_result': avg_noisy, 'mae': mae, 'avg_time_ms': avg_time * 1000})
            print(f"  Epsilon {eps}: Avg Noisy={avg_noisy:.2f}, MAE={mae:.2f}, Time={(avg_time*1000):.4f}ms")

        # Sum
        sum_data_list = df[sum_column].astype(float).tolist()
        print(f"\nTesting Sum Query for {dataset_name} (Google DP):")
        for eps in epsilon_values:
            noisy_results = []
            start_time = time.time()
            try:
                google_sum_runner = GoogleDPBoundedSumLaplace(epsilon=eps, lower_bound=sum_lower_bound, upper_bound=sum_upper_bound)
                for _ in range(num_runs):
                    noisy_results.append(google_sum_runner.run(sum_data_list))
            except Exception as e:
                print(f"  Epsilon {eps}: Error - {e}")
                current_dataset_results['sum_laplace'].append({'epsilon': eps, 'avg_noisy_result': 'Error', 'mae': 'Error', 'avg_time_ms': 'Error'})
                continue
            avg_time = (time.time() - start_time) / num_runs
            avg_noisy = np.mean(noisy_results)
            mae = np.mean(np.abs(np.array(noisy_results) - true_sum))
            current_dataset_results['sum_laplace'].append({'epsilon': eps, 'avg_noisy_result': avg_noisy, 'mae': mae, 'avg_time_ms': avg_time * 1000})
            print(f"  Epsilon {eps}: Avg Noisy={avg_noisy:.2f}, MAE={mae:.2f}, Time={(avg_time*1000):.4f}ms")
        
        all_results_summary[dataset_name] = current_dataset_results
    return all_results_summary
