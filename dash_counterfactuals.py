import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import base64
import io
from dash_ag_grid import AgGrid

# Initialize the Dash app
app = dash.Dash(__name__, requests_pathname_prefix='/Reasoning/CounterfactualsDash/', suppress_callback_exceptions=True)

# Layout of the application
app.layout = html.Div([
    # Upload Dataset Section
    html.H3("Upload Dataset"),
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Button('Upload File', className='btn-upload'),
            multiple=False  # Only allow one file
        )
    ], className="upload-container"),

    html.Br(),

    # Table of predictor variables
    html.Div([
        html.H3("Predictor Variables"),
        AgGrid(
            id='predictor-table',
            columnDefs=[],  # Column Definitions will be filled after file upload
            rowData=[],  # Data will be filled after file upload
            defaultColDef={'editable': False, 'resizable': True, 'sortable': True},  # Set default column behavior
            dashGridOptions={'rowSelection': 'single'},  # Enable single row selection
            style={'height': '300px'}  # We'll dynamically set the width later
        )
    ], id='predictor-container', style={'display': 'none'}),  # Hide until file is uploaded

    html.Br(),

    # Section to display the selected row and modify class
    html.Div([
        html.H3("Selected Row"),
        AgGrid(
            id='selected-row-table',
            columnDefs=[],  # Column Definitions for selected row
            rowData=[],  # Data for the selected row
            defaultColDef={'editable': False, 'resizable': True},  # Set default column behavior
            style={'height': '100px', 'width': '100%'}
        )
    ], id='selected-row-container', style={'display': 'none'}),  # Hide until a row is selected

    # Dropdown for modifying the class of the selected row
    html.Div([
        html.H3("Select Class"),
        dcc.Dropdown(id='class-selector')  # Dropdown to select class
    ], id='class-container', style={'display': 'none'}),  # Hide until a row is selected

    html.Br(),

    # Dropdown for number of models and Run button
    html.Div([
        html.H3("Number of Models"),
        dcc.Dropdown(
            id='model-selector',
            options=[{'label': str(i), 'value': i} for i in range(1, 6)],
            value=5  # Default to 5 models
        ),
        html.Br(),
        html.Div([
            html.Button('Run', id='run-button')
        ], className="run-container")
    ], id='model-container', style={'display': 'none'}),  # Hide until a row is selected

    html.Br(),

    # Results Table
    html.Div([
        html.H3("Results"),
        AgGrid(
            id='results-table',
            columnDefs=[],  # Column Definitions for results
            rowData=[],  # Data for the results
            defaultColDef={'resizable': True},  # Make columns resizable
            style={'height': '200px'}  # Dynamically handle the width later
        )
    ], id='results-container', style={'display': 'none'})  # Hide until results are generated
])

# Function to parse uploaded CSV or Excel file
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        # Reset the index to avoid including it in rowData
        df = df.reset_index(drop=True)
        return df
    except Exception as e:
        print(e)
        return None

# Callback to update the predictor table and show it after file upload
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
        # Only include columns with data and avoid any extra or index columns
        columns = [{'headerName': i, 'field': i, 'width': 200} for i in df.columns]
        data = df.to_dict('records')
        # Calculate total grid width based on column widths
        total_width = sum([col['width'] for col in columns])
        return data, columns, {'height': '300px', 'width': f'{total_width}px'}, {'display': 'block'}
    return [], [], {'height': '300px', 'width': '100%'}, {'display': 'none'}

# Callback to handle row selection and update class dropdown
@app.callback(
    [Output('selected-row-table', 'rowData'),
     Output('selected-row-table', 'columnDefs'),
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
        selected_row = selectedRows[0]  # This is already a dictionary
        row_data = [selected_row]
        columns = [{'headerName': i, 'field': i} for i in selected_row.keys()]

        # Extract class options for dropdown
        if 'class' in selected_row:
            class_options = [{'label': cls, 'value': cls} for cls in set([row['class'] for row in data])]
            class_value = selected_row['class']
            return row_data, columns, {'display': 'block'}, class_options, class_value, {'display': 'block'}, {'display': 'block'}

    return [], [], {'display': 'none'}, [], None, {'display': 'none'}, {'display': 'none'}

# Callback to handle the Run button and generate results
@app.callback(
    [Output('results-table', 'rowData'),
     Output('results-table', 'columnDefs'),
     Output('results-container', 'style')],
    Input('run-button', 'n_clicks'),
    State('predictor-table', 'selectedRows'),
    State('class-selector', 'value'),
    State('model-selector', 'value')
)
def run_counterfactual(n_clicks, selectedRows, new_class, num_models):
    if n_clicks is None or not selectedRows:
        return [], [], {'display': 'none'}

    selected_row = selectedRows[0]  # Already a dict
    row_data = selected_row.copy()

    # Logic to generate counterfactual (simplified)
    original_row = row_data.copy()
    new_row = row_data.copy()
    new_row['class'] = new_class

    results_data = [original_row, new_row]
    results_columns = [{'headerName': i, 'field': i} for i in original_row.keys()]
    return results_data, results_columns, {'display': 'block'}

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
