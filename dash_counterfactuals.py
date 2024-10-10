import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import base64
import io
from dash_ag_grid import AgGrid
import numpy as np
from sklearn.model_selection import train_test_split
from ensemble_counterfactuals.common_funcs import train_models
from ensemble_counterfactuals.algorithms import ga, eda, moeda, nsga2, ebna, moebna
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter

# Activate the pandas2ri conversion globally
pandas2ri.activate()

# Initialize the Dash app
app = dash.Dash(
    __name__,
    requests_pathname_prefix='/Reasoning/CounterfactualsDash/',
    suppress_callback_exceptions=True
)

# Global variable to store the uploaded DataFrame
uploaded_df = pd.DataFrame()

# Layout of the application
app.layout = html.Div([
    # Upload Dataset Section
    html.H3("Upload Dataset", style={'textAlign': 'center'}),
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Button('Upload File', id='upload-button'),
            multiple=False  # Only allow one file
        )
    ], style={'textAlign': 'center'}),

    html.Br(),

    # Table of predictor variables
    html.Div([
        html.H3("Predictor Variables", style={'textAlign': 'center'}),
        html.Div([
            AgGrid(
                id='predictor-table',
                columnDefs=[],  # Column Definitions will be filled after file upload
                rowData=[],     # Data will be filled after file upload
                defaultColDef={'editable': False, 'resizable': True, 'sortable': True},
                dashGridOptions={'rowSelection': 'single'},  # Enable single row selection
                style={'height': '300px'}  # We'll dynamically set the width later
            )
        ], style={'display': 'flex', 'justifyContent': 'center'})
    ], id='predictor-container', style={'display': 'none'}),

    html.Br(),

    # Selected Row and Class Modification Section
    html.Div([
        html.H3("Selected Row", style={'textAlign': 'center'}),
        html.Div([
            AgGrid(
                id='selected-row-table',
                columnDefs=[],
                rowData=[],
                defaultColDef={'editable': False, 'resizable': True}
            )
        ], style={'display': 'flex', 'justifyContent': 'center'})
    ], id='selected-row-container', style={'display': 'none'}),

    html.Div([
        html.H3("Select Class", style={'textAlign': 'center'}),
        html.Div([
            dcc.Dropdown(id='class-selector')
        ], style={'width': '200px', 'margin': '0 auto'})
    ], id='class-container', style={'display': 'none'}),

    html.Br(),

    # Number of Models and Run Button
    html.Div([
        html.H3("Number of Models", style={'textAlign': 'center'}),
        html.Div([
            dcc.Dropdown(
                id='model-selector',
                options=[{'label': str(i), 'value': i} for i in range(1, 6)],
                value=5
            )
        ], style={'width': '200px', 'margin': '0 auto'}),
        html.Br(),
        html.Div([
            html.Button('Run', id='run-button')
        ], style={'textAlign': 'center'})
    ], id='model-container', style={'display': 'none'}),

    html.Br(),

    # Results Table
    html.Div([
        html.H3("Results", style={'textAlign': 'center'}),
        html.Div([
            AgGrid(
                id='results-table',
                columnDefs=[],
                rowData=[],
                defaultColDef={'resizable': True}
            )
        ], style={'display': 'flex', 'justifyContent': 'center'})
    ], id='results-container', style={'display': 'none'})
])

# Function to parse uploaded CSV or Data file
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    global uploaded_df
    try:
        if any(ext in filename.lower() for ext in ['csv', 'data']):
            # Assuming the .data file is comma-separated and may not have headers
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), header=None)
            # Provide column names if needed
            df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        elif 'xls' in filename.lower() or 'xlsx' in filename.lower():
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None
        df = df.reset_index(drop=True)
        uploaded_df = df.copy()  # Store the DataFrame globally
        return df
    except Exception as e:
        print(f"Error parsing file: {e}")
        return None


# Callback to update the predictor table
@app.callback(
    [Output('predictor-table', 'rowData'),
     Output('predictor-table', 'columnDefs'),
     Output('predictor-table', 'style'),
     Output('predictor-container', 'style')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_predictor_table(contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        if df is None:
            return [], [], {'height': '400px', 'width': '100%'}, {'display': 'none'}
        columns = [{'headerName': col, 'field': col, 'width': 200} for col in df.columns]
        data = df.to_dict('records')
        total_width = sum([col['width'] for col in columns])
        return data, columns, {'height': '400px', 'width': f'{total_width}px'}, {'display': 'block'}
    else:
        return [], [], {'height': '400px', 'width': '100%'}, {'display': 'none'}

# Callback to display selected row and update class options
@app.callback(
    [Output('selected-row-table', 'rowData'),
     Output('selected-row-table', 'columnDefs'),
     Output('selected-row-table', 'style'),
     Output('selected-row-container', 'style'),
     Output('class-selector', 'options'),
     Output('class-selector', 'value'),
     Output('class-container', 'style'),
     Output('model-container', 'style')],
    Input('predictor-table', 'selectedRows'),
    State('predictor-table', 'rowData')
)
def display_selected_row_and_class(selectedRows, data):
    if selectedRows:
        selected_row = selectedRows[0]
        # Exclude internal keys
        row_data = [{k: v for k, v in selected_row.items() if not k.startswith('_')}]
        columns = [{'headerName': col, 'field': col, 'width': 200} for col in row_data[0].keys()]
        total_width = sum([col['width'] for col in columns])

        # Extract class options for dropdown
        if 'class' in selected_row:
            class_options = [{'label': cls, 'value': cls} for cls in sorted({row['class'] for row in data})]
            class_value = selected_row['class']
            return (
                row_data,
                columns,
                {'height': '100px', 'width': f'{total_width}px'},
                {'display': 'block'},
                class_options,
                class_value,
                {'display': 'block'},
                {'display': 'block'}
            )
    return [], [], {}, {'display': 'none'}, [], None, {'display': 'none'}, {'display': 'none'}


# Callback para ejecutar la generación de contrafactuales
@app.callback(
    [Output('results-table', 'rowData'),
     Output('results-table', 'columnDefs'),
     Output('results-table', 'style'),
     Output('results-container', 'style')],
    Input('run-button', 'n_clicks'),
    State('predictor-table', 'selectedRows'),
    State('model-selector', 'value'),
    State('class-selector', 'value')
)
def run_counterfactual(n_clicks, selectedRows, num_models, new_class):
    if n_clicks is None or not selectedRows or new_class is None:
        return [], [], {}, {'display': 'none'}

    try:
        selected_row = selectedRows[0]
        selected_row_clean = {k: v for k, v in selected_row.items() if not k.startswith('_')}

        # Generate counterfactuals
        df_counterfactual = generate_counterfactuals(selected_row_clean, new_class, num_models, uploaded_df)
        print(f"df_counterfactual:\n{df_counterfactual}")
        if df_counterfactual is not None and not df_counterfactual.empty:
            # Convert the pandas DataFrame to R DataFrame
            r_from_pd_df = robjects.conversion.py2rpy(df_counterfactual)
            robjects.globalenv['r_from_pd_df'] = r_from_pd_df  # Assign to global environment in R

            data = df_counterfactual.to_dict('records')
            columns = [{'headerName': col, 'field': col, 'width': 200} for col in df_counterfactual.columns]
            total_width = sum([col['width'] for col in columns])
            return data, columns, {'height': '300px', 'width': f'{total_width}px'}, {'display': 'block'}
        else:
            print("No counterfactuals to display")
            return [], [], {}, {'display': 'none'}
    except Exception as e:
        # Log the error
        print(f"Error generating counterfactuals: {e}")
        # Optionally, display an error message in the interface
        return [], [], {}, {'display': 'none'}

def preprocess_data(df):
    # Handle missing values
    df = df.dropna()

    # Convert all columns to 'category' dtype
    for col in df.columns:
        df[col] = df[col].astype('category')

    # All columns are categorical
    categorical_columns = df.columns.tolist()

    return df, categorical_columns


def determine_discrete_variables(df):
    discrete_vars = [True] * (df.shape[1] - 1)  # Exclude 'class' column
    return discrete_vars


def generate_counterfactuals(selected_row, new_class, num_models, df):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    # Preprocess data
    df, categorical_columns = preprocess_data(df)

    # Perform train-test split
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    # Convert selected_row to DataFrame
    selected_row_df = pd.DataFrame([selected_row])

    # Ensure consistent data types and factor levels
    for col in df.columns:
        selected_row_df[col] = selected_row_df[col].astype('category')
        selected_row_df[col].cat.set_categories(df[col].cat.categories)

    # Ensure consistent levels for categorical variables
    for col in categorical_columns:
        categories = df[col].cat.categories  # Get categories from the full dataset
        train_df[col] = train_df[col].cat.set_categories(categories)
        test_df[col] = test_df[col].cat.set_categories(categories)
        selected_row_df[col] = selected_row_df[col].astype('category')
        selected_row_df[col] = selected_row_df[col].cat.set_categories(categories)

        
    # Combine X_train and y_train
    df_train = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Check for mismatched levels in the input instance
    for col in selected_row_df.columns:
        input_value = selected_row_df[col].iloc[0]
        if input_value not in df_train[col].cat.categories:
            print(f"Value '{input_value}' in column '{col}' is not in training data levels: {df_train[col].cat.categories.tolist()}")
            return None

    # Print levels in training data
    print("Levels in training data:")
    for col in df_train.columns:
        levels = df_train[col].cat.categories.tolist()
        print(f"Levels for '{col}': {levels}")

    # Print values in input instance
    print("Values in input instance:")
    for col in selected_row_df.columns:
        value = selected_row_df[col].iloc[0]
        print(f"'{col}': '{value}'")

    # Determine discrete variables
    discrete_variables = [True] * (df_train.shape[1] - 1)  # Exclude 'class'

    # Map new_class to its original label
    obj_class_label = new_class

    # Before ensemble_counter_eda call
    print("Starting ensemble_counter_eda")
    print(f"Input instance: {selected_row_df.iloc[0].values}")
    print(f"Objective class: {obj_class_label}")
    print(f"Number of models: {num_models}")
    print(f"Discrete variables: {discrete_variables}")

    try:
        # Comprobación de niveles en los DataFrames antes de pasarlos a R
        print("Niveles en train_df:")
        print(train_df['class'].cat.categories)
        print("Niveles en test_df:")
        print(test_df['class'].cat.categories)

        # Verificar las primeras filas de cada DataFrame
        print("Primeras filas de train_df:")
        print(train_df.head())
        print("Primeras filas de test_df:")
        print(test_df.head())

        # Adjust the call to ensemble_counter_eda
        df_result, _, accuracy, time_taken = eda.ensemble_counter_eda(
            X=df_train,
            input=selected_row_df.iloc[0].values,
            obj_class=new_class,
            test=df_train,
            discrete_variables=discrete_variables,
            verbose=True,
            no_train=True
        )
        print("ensemble_counter_eda completed")
    except Exception as e:
        print(f"Error in ensemble_counter_eda: {e}")
        return None

    if df_result is not None and not df_result.empty:
        print("Counterfactuals generated successfully")
        print(df_result)
        return df_result
    else:
        print("No counterfactuals were generated")
        return None



def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    global uploaded_df
    try:
        if any(ext in filename.lower() for ext in ['csv', 'data']):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), header=None, dtype=str)
            df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        elif 'xls' in filename.lower() or 'xlsx' in filename.lower():
            df = pd.read_excel(io.BytesIO(decoded), dtype=str)
        else:
            return None
        df = df.reset_index(drop=True)
        uploaded_df = df.copy()
        return df
    except Exception as e:
        print(f"Error parsing file: {e}")
        return None


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)