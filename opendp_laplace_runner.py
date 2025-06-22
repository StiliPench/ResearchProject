import pandas as pd
import numpy as np
import os
import time
import opendp
from opendp.mod import enable_features
from opendp.domains import atom_domain, vector_domain, Domain
from opendp.metrics import symmetric_distance, Metric
from opendp.measurements import make_laplace, Measurement
from opendp.transformations import make_count, make_bounded_float_ordered_sum, Transformation
from opendp.typing import f64

enable_features("floating-point", "contrib")

# configurations
DEFAULT_EPSILON_VALUES = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
DEFAULT_NUM_RUNS = 100
DEFAULT_SUM_LOWER_BOUND = -100.0
DEFAULT_SUM_UPPER_BOUND = 100.0
DEFAULT_SUM_COLUMN = 'value_col_1'

def build_count_laplace_measurement(epsilon: float) -> Measurement:
    input_domain: Domain = vector_domain(atom_domain(T=f64))
    input_metric: Metric = symmetric_distance()
    count_transform: Transformation = make_count(input_domain=input_domain, input_metric=input_metric)
    laplace_scale = 1.0 / epsilon
    laplace_addition: Measurement = make_laplace(
        input_domain=count_transform.output_domain,
        input_metric=count_transform.output_metric,
        scale=laplace_scale
    )
    return count_transform >> laplace_addition

def build_bounded_sum_laplace_measurement(epsilon: float, lower: float, upper: float, size_limit_for_sum: int) -> Measurement:
    sum_transform: Transformation = make_bounded_float_ordered_sum(
        size_limit=size_limit_for_sum, bounds=(lower, upper), S=f"Pairwise<{f64}>"
    )
    sum_l1_sensitivity = max(abs(lower), abs(upper))
    laplace_scale = sum_l1_sensitivity / epsilon
    laplace_addition: Measurement = make_laplace(
        input_domain=sum_transform.output_domain,
        input_metric=sum_transform.output_metric,
        scale=laplace_scale
    )
    return sum_transform >> laplace_addition

def run_opendp_laplace_experiments(datasets_to_test,
                                   epsilon_values=DEFAULT_EPSILON_VALUES,
                                   num_runs=DEFAULT_NUM_RUNS,
                                   sum_lower_bound=DEFAULT_SUM_LOWER_BOUND,
                                   sum_upper_bound=DEFAULT_SUM_UPPER_BOUND,
                                   sum_column=DEFAULT_SUM_COLUMN):
    all_results_summary = {}

    for ds_info in datasets_to_test:
        dataset_name = ds_info['name']
        dataset_file = ds_info['file']
        print(f"\n\n--- Processing Dataset (OpenDP): {dataset_name} ({dataset_file}) ---")
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
        
        # Count
        first_column_name = df.columns[0]
        data_for_count_list = df[first_column_name].astype(float).tolist()
        print(f"\nTesting Count Query for {dataset_name} (OpenDP):")
        for eps in epsilon_values:
            noisy_results = []
            start_time = time.time()
            try:
                meas = build_count_laplace_measurement(epsilon=eps)
                for _ in range(num_runs):
                    noisy_results.append(meas(data_for_count_list))
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
        sum_data_list = df[sum_column].tolist()
        print(f"\nTesting Sum Query for {dataset_name} (OpenDP):")
        for eps in epsilon_values:
            noisy_results = []
            start_time = time.time()
            try:
                meas = build_bounded_sum_laplace_measurement(epsilon=eps, lower=sum_lower_bound, upper=sum_upper_bound, size_limit_for_sum=true_count)
                for _ in range(num_runs):
                    noisy_results.append(meas(sum_data_list))
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
