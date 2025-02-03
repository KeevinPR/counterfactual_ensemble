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
app.layout = dcc.Loading(
    id="global-spinner",
    overlay_style={"visibility":"visible", "filter": "blur(1px)"},
    type="circle",        # You can choose "circle", "dot", "default", etc.
    fullscreen=False,      # This ensures it covers the entire page
    children=html.Div([
    # Upload Dataset Section
    html.H1("Counterfactuals", style={'textAlign': 'center'}),
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
    # Number of Models and Run Button
    html.Div([
        html.H3("Models", style={'textAlign': 'center'}),
        html.Div([
            html.P("nb, tn, fssj, kdb, tanhc, baseline", style={'textAlign': 'center', 'fontSize': '18px'}),
            html.P("5 models will be used", style={'textAlign': 'center', 'fontSize': '12px'})
        ], style={'width': '200px', 'margin': '0 auto'}),
        html.Br(),
        dcc.Loading(
            id='loading-run-button',
            type='circle',
            children=[
                html.Div([
                    html.Button('Run', id='run-button', n_clicks=0)
                ], style={'textAlign': 'center'}),
                dcc.Store(id='run-button-store')
            ]
        )
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
    ], id='results-container', style={'display': 'none'}),
    #Div for automatic srolling down
    html.Div(id='scroll-helper', style={'display': 'none'})
])
)
#Automix scrolling down
app.clientside_callback(
    """
    function(selectedRowStyle, resultsStyle) {
        // Verificar si el contenedor de 'Selected Row' está visible
        if (selectedRowStyle && selectedRowStyle.display === 'block') {
            document.getElementById('selected-row-container').scrollIntoView({behavior: 'smooth'});
        }
        // Verificar si el contenedor de 'Results' está visible
        else if (resultsStyle && resultsStyle.display === 'block') {
            document.getElementById('results-container').scrollIntoView({behavior: 'smooth'});
        }
        return '';
    }
    """,
    Output('scroll-helper', 'children'),
    [Input('selected-row-container', 'style'),
     Input('results-container', 'style')]
)

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
        
        # Reset index to create 'Row Number' column
        df = df.reset_index(drop=False)
        df.rename(columns={'index': 'Row Number'}, inplace=True)
        
        columns = [{'headerName': col, 'field': col, 'width': 120} for col in df.columns]
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
        # Exclude internal keys (starting with '_')
        row_data = [{k: v for k, v in selected_row.items() if not k.startswith('_')}]
        columns = [{'headerName': col, 'field': col, 'width': 120} for col in row_data[0].keys()]
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



# Callback para ejecutar la generación de contrafactuales con mejoras
# Add 'run-button.disabled' and 'run-button-store.data' to the Outputs
@app.callback(
    [Output('results-table', 'rowData'),
     Output('results-table', 'columnDefs'),
     Output('results-table', 'style'),
     Output('results-container', 'style'),
     Output('run-button', 'disabled'),
     Output('run-button-store', 'data')],
    Input('run-button', 'n_clicks'),
    State('predictor-table', 'selectedRows'),
    State('class-selector', 'value'),
    State('upload-data', 'contents')
)
def run_counterfactual(n_clicks, selectedRows, new_class, contents):
    num_models = 5
    if n_clicks is None or n_clicks == 0 or not selectedRows or new_class is None or contents is None:
        return [], [], {}, {'display': 'none'}, False, None  # Button enabled
    
    # Disable the "Run" button during processing
    disabled = True
    
    try:
        # Parse the uploaded data
        filename = 'temp_data.csv'  # Temporary filename
        df = parse_contents(contents, filename)
    
        # Check if data loaded correctly
        if df is None or df.empty:
            print("Error: The data was not loaded correctly.")
            return [], [], {}, {'display': 'none'}, False, None  # Button enabled
    
        # Process the selected row
        selected_row = selectedRows[0]
        # Exclude internal keys and 'Row Number'
        selected_row_clean = {k: v for k, v in selected_row.items() if not k.startswith('_') and k != 'Row Number'}
    
        # Validate input levels
        if not validate_input_levels(df, selected_row_clean):
            print("Error: Levels in the selected instance do not match levels in the loaded data.")
            return [], [], {}, {'display': 'none'}, False, None  # Button enabled
    
        # Generate counterfactuals
        df_counterfactual = generate_counterfactuals(selected_row_clean, new_class, num_models, df)
        #print(f"df_counterfactual:\n{df_counterfactual}")
    
        # Check if counterfactuals were generated
        if df_counterfactual is not None and not df_counterfactual.empty:
            # Prepare the results table
            data = df_counterfactual.to_dict('records')
            columns = [{'headerName': col, 'field': col, 'width': 100,} for col in df_counterfactual.columns]
            total_width = sum([col['width'] for col in columns])
            # Re-enable the "Run" button and update the store
            disabled = False
            return data, columns, {'height': '300px', 'width': f'{total_width}px'}, {'display': 'block'}, disabled, 'done'
        else:
            #print("No counterfactuals to display")
            # Re-enable the "Run" button and update the store
            disabled = False
            return [], [], {}, {'display': 'none'}, disabled, 'done'
    except Exception as e:
        # Log the error
        print(f"Error generating counterfactuals: {e}")
        # Re-enable the "Run" button and update the store
        disabled = False
        return [], [], {}, {'display': 'none'}, disabled, 'done'




def validate_input_levels(df, selected_row):
    """
    Función para validar si los niveles en la instancia seleccionada coinciden con los niveles en el DataFrame cargado.
    """
    for col in selected_row:
        if col in df.columns and selected_row[col] not in df[col].unique():
            #print(f"Valor '{selected_row[col]}' en columna '{col}' no coincide con los niveles de los datos cargados.")
            return False
    return True

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

    # Preprocess data to ensure all columns are treated as categories
    df, categorical_columns = preprocess_data(df)

    # Perform train-test split
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    # Convert selected_row to DataFrame
    selected_row_df = pd.DataFrame([selected_row])

    # Ensure consistent data types and factor levels
    for col in df.columns:
        selected_row_df[col] = selected_row_df[col].astype('category')
        selected_row_df[col] = selected_row_df[col].cat.set_categories(df[col].cat.categories)

    # Ensure consistent levels for categorical variables in train_df and test_df
    for col in categorical_columns:
        categories = df[col].cat.categories  # Get categories from the full dataset
        train_df[col] = train_df[col].cat.set_categories(categories)
        test_df[col] = test_df[col].cat.set_categories(categories)
        selected_row_df[col] = selected_row_df[col].cat.set_categories(categories)

    # Convert data to strings for compatibility with R code
    train_df = train_df.astype(str)
    test_df = test_df.astype(str)
    selected_row_df = selected_row_df.astype(str)

    # Determine discrete variables
    discrete_variables = [True] * (df.shape[1] - 1)  # Exclude 'class'

    # Map new_class to its original label
    obj_class_label = new_class

    # Before ensemble_counter_eda call
    #print("Starting ensemble_counter_eda")
    #print(f"Input instance: {selected_row_df.iloc[0].values}")
    #print(f"Objective class: {obj_class_label}")
    #print(f"Number of models: {num_models}")
    #print(f"Discrete variables: {discrete_variables}")

    try:
        # Llamada a ensemble_counter_eda con datos de tipo string
        df_result, _, accuracy, time_taken = eda.ensemble_counter_eda(
            X=train_df,
            input=selected_row_df.iloc[0].values,
            obj_class=new_class,
            test=test_df,
            discrete_variables=discrete_variables,
            verbose=True,
            no_train=True
        )
        #print("ensemble_counter_eda completed")
    except Exception as e:
        print(f"Error in ensemble_counter_eda: {e}")
        return None

    if df_result is not None and not df_result.empty:
        #print("Counterfactuals generated successfully")
        return df_result
    else:
        #print("No counterfactuals were generated")
        return None


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    global uploaded_df
    try:
        if any(ext in filename.lower() for ext in ['csv', 'data']):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), dtype=str)
            #df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), header=None, dtype=str)
            #df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        elif 'xls' in filename.lower() or 'xlsx' in filename.lower():
            df = pd.read_excel(io.BytesIO(decoded), dtype=str)
        else:
            return None
        df = df.reset_index(drop=True)
        uploaded_df = df.copy()
        return df
    #except pd.errors.ParserError:
        # Si falla, intentar leer sin encabezados
        #df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), header=None, dtype=str)
        # Asignar nombres genéricos a las columnas
        #df.columns = [f'col_{i}' for i in range(df.shape[1])]
    except Exception as e:
        print(f"Error parsing file: {e}")
        return None


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)