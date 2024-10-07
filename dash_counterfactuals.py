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

    selected_row = selectedRows[0]
    selected_row_clean = {k: v for k, v in selected_row.items() if not k.startswith('_')}

    # Generate counterfactuals
    df_counterfactual = generate_counterfactuals(selected_row_clean, new_class, num_models, uploaded_df)

    if df_counterfactual is not None:
        # Convert the pandas DataFrame to R DataFrame
        r_from_pd_df = robjects.conversion.py2rpy(df_counterfactual)
        robjects.globalenv['r_from_pd_df'] = r_from_pd_df  # Assign to global environment in R

        data = df_counterfactual.to_dict('records')
        columns = [{'headerName': col, 'field': col, 'width': 200} for col in df_counterfactual.columns]
        total_width = sum([col['width'] for col in columns])
        return data, columns, {'height': '300px', 'width': f'{total_width}px'}, {'display': 'block'}
    else:
        return [], [], {}, {'display': 'none'}

def generate_counterfactuals(selected_row, new_class, num_models, df):
    from rpy2.robjects import pandas2ri, default_converter
    from rpy2.robjects.conversion import localconverter
    
    # Activate pandas2ri conversion and set the conversion in this thread
    pandas2ri.activate()
    robjects.conversion.set_conversion(default_converter + pandas2ri.converter)
    
    if 'class' not in df.columns:
        raise ValueError("The dataframe does not contain a 'class' column.")

    # Prepare data
    X = df.drop(columns=['class']).copy()
    y = df['class'].copy()
    selected_row_df = pd.DataFrame([selected_row])[X.columns]
    X_train = X[~(X == selected_row_df.iloc[0]).all(axis=1)]
    test_df = selected_row_df.copy()

    # Combine X_train and y to create the training DataFrame
    df_train = X_train.copy()
    df_train['class'] = y.loc[X_train.index]

    # Rename 'class' to 'class_label' for R
    df_train_for_R = df_train.rename(columns={'class': 'class_label'})
    test_df_for_R = test_df.copy()
    test_df_for_R['class_label'] = new_class

    # Print the type of robjects.conversion for debugging
    print("dash Before train_models, type of robjects.conversion:", type(robjects.conversion))


    # Convert pandas DataFrames to R DataFrames using localconverter
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_df_train_for_R = robjects.conversion.py2rpy(df_train_for_R)
        r_test_df_for_R = robjects.conversion.py2rpy(test_df_for_R)


    robjects.globalenv['r_df_train_for_R'] = r_df_train_for_R
    robjects.globalenv['r_test_df_for_R'] = r_test_df_for_R

    # Train models
    train_models(df_train_for_R, test_df_for_R)
    print("dash After train_models, type of robjects.conversion:", type(robjects.conversion))

    # Generate counterfactuals
    input_instance = test_df.iloc[0].values
    df_result, _, accuracy, time_taken = eda.ensemble_counter_eda(
        X=df_train_for_R,
        input=input_instance,
        obj_class=new_class,
        test=test_df_for_R,
        discrete_variables=[True] * X.shape[1],
        verbose=False,
        no_train=True
    )

    if df_result is not None:
        return df_result
    else:
        return None

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)