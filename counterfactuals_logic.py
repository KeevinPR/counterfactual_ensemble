# counterfactuals_logic.py

# Necessary imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statistics import mean 
from ensemble_counterfactuals.common_funcs import train_models
from ensemble_counterfactuals.algorithms import ga, eda, moeda, nsga2, ebna, moebna

def generate_counterfactuals(selected_row, new_class, num_models, df):
    """
    Generates counterfactuals based on the provided data.

    :param selected_row: The data of the selected row (as a dictionary).
    :param new_class: The new class assigned by the user.
    :param num_models: Number of models selected to generate the counterfactuals.
    :param df: The complete DataFrame with the data.

    :return: DataFrame containing the original and counterfactual rows.
    """
    # Check if 'class' column exists
    if 'class' not in df.columns:
        raise ValueError("The data does not contain a 'class' column.")

    # Prepare the data
    X = df.drop(columns=['class']).copy()
    y = df['class'].copy()
    X['class'] = y

    # Get unique class values
    class_values = y.unique()
    one_sol = False  # This variable is not used further in the example code

    # Assuming all variables are discrete; adjust if needed
    discrete_variables = [True] * (X.shape[1] - 1)

    # Convert selected_row to DataFrame and ensure column order matches
    selected_row_df = pd.DataFrame([selected_row])[X.columns]

    # Remove the selected instance from the data to prevent duplication
    X_train = X[~(X == selected_row_df.iloc[0]).all(axis=1)]

    # Use the selected row as the test set
    test_df = selected_row_df.copy()

    # Train the models with the training and test data
    train_models(X_train, test_df)

    # Execution example
    instance_number = 0  # Since test_df has only one instance
    obj_class = new_class  # Use the new class selected by the user

    results = []
    dfs = []
    times = []
    print(f'Instance number: {instance_number}')

    # Get the input instance
    input_instance = test_df.iloc[instance_number].values

    # Remove the current class from class_values
    cls_val = np.delete(class_values, np.where(class_values == test_df.iloc[instance_number]['class']))

    print(test_df.iloc[instance_number])

    # Generate counterfactuals using the EDA algorithm
    # Adjust the algorithm used here if needed
    df_result, rem_models, accuracy, time_taken = eda.ensemble_counter_eda(
        X=X_train,
        input=input_instance,
        obj_class=obj_class,
        test=test_df,
        discrete_variables=discrete_variables,
        verbose=False,
        no_train=True
    )

    if df_result is not None:
        times.append(time_taken)
        dfs.append(df_result)
        models = {}
        chg_variables = {}
        prediction = {}
        plausible = {}

        for e in df_result["model"].unique():
            if e != "input":
                models[e] = []
                if e != "baseline":
                    chg_variables[e] = []
                    prediction[e] = []
                    plausible[e] = []

        for row in df_result.values:
            if row[0] == "input":
                continue
            models[row[0]].append(float(row[-2]))
            if row[0] != "baseline":
                chg_variables[row[0]].append(float(row[-4]))
                prediction[row[0]].append(float(row[-3]))
                plausible[row[0]].append(float(row[-1]))

        baseline = models['baseline'][0]
        model_distances = {}
        for model_name in models.keys():
            if model_name in ["input", "baseline"]:
                continue
            dict_dist = {"distance": 1e12, "baseline": None, 'distance_ind': 0}
            for ind, elem in enumerate(models[model_name]):
                if dict_dist["distance"] >= elem:
                    dict_dist["distance"] = elem
                    dict_dist["distance_ind"] = ind
                if elem == baseline:
                    dict_dist["baseline"] = elem
                    dict_dist["baseline_ind"] = ind
            dict_dist["chg_variables"] = chg_variables[model_name][dict_dist["distance_ind"]]
            dict_dist["prediction"] = prediction[model_name][dict_dist["distance_ind"]]
            dict_dist["plausible"] = plausible[model_name][dict_dist["distance_ind"]]
            dict_dist["accuracy"] = accuracy[rem_models.index(model_name)]
            model_distances[model_name] = dict_dist
        results.append(model_distances)

        print(f'Mean time: {mean(times)}')
        print('Results:', results)
        print('Objective class:', obj_class)

        # Return the result DataFrame
        return df_result
    else:
        return None
