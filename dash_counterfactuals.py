import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
from dash import dash_table
import base64
import io

# Inicialización de la app con el prefijo de ruta
app = dash.Dash(__name__, requests_pathname_prefix='/Reasoning/CounterfactualsDash/', suppress_callback_exceptions=True)

# Layout de la aplicación
app.layout = html.Div([
    # Componente para subir el archivo
    html.H3("Upload Dataset"),
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Button('Upload File', className='btn-upload'),  # Añadir clase al botón
            multiple=False  # Solo permitimos un archivo a la vez
        )
    ], className="upload-container"),  # Centrar el botón con la clase 'upload-container'

    html.Br(),

    # Contenedor para la tabla de variables predictoras y su título
    html.Div([
        html.H3("Predictor Variables"),
        dash_table.DataTable(
            id='predictor-table',
            columns=[],  # Inicialmente vacío
            data=[],  # Inicialmente vacío
            page_size=10,
            row_selectable='single',  # Selección de una fila
            selected_rows=[]  # Filas seleccionadas
        ),
    ], id='predictor-container', style={'display': 'none'}),  # Ocultar el contenedor hasta que se cargue el archivo

    html.Br(),

    # Input - Mostrar fila seleccionada
    html.Div([
        html.H3("Selected Row"),
        dash_table.DataTable(
            id='selected-row-table',  # Tabla para mostrar la fila seleccionada
            columns=[],  # Inicialmente vacío
            data=[],  # Inicialmente vacío
            page_size=1,  # Mostrar solo la fila seleccionada
        )
    ], style={'display': 'none'}, id='selected-row-container'),  # Ocultamos la tabla hasta que se seleccione una fila

    # Contenedor del Dropdown para seleccionar la clase
    html.Div([
        html.Label("Class:"),
        dcc.Dropdown(id='class-selector')
    ], id='class-container', style={'display': 'none'}),  # Oculto hasta que se selecciona una fila

    # Contenedor para el selector de modelos y el botón Run
    html.Div([
        html.H3("Number of Models"),
        dcc.Dropdown(
            id='model-selector',
            options=[{'label': str(i), 'value': i} for i in range(1, 6)],
            value=5  # Valor por defecto
        ),
        html.Br(),

        # Botón de Run, ahora con la clase para centrarlo y aplicar estilo
        html.Div([
            html.Button('Run', id='run-button')
        ], className="run-container"),  # Centrar el botón con la clase 'run-container'
    ], id='model-container', style={'display': 'none'}),  # Oculto hasta que se seleccione una fila

    html.Br(),

    # Contenedor para los resultados
    html.Div([
        html.H3("Results"),
        dash_table.DataTable(
            id='results-table',
            columns=[],  # Inicialmente vacío
            data=[]  # Inicialmente vacío
        )
    ], id='results-container', style={'display': 'none'})  # Oculto hasta que se generen los resultados
])


# Función para parsear el archivo CSV subido
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        return df
    except Exception as e:
        print(e)
        return None

# Callback para actualizar la tabla de variables predictoras y mostrar el contenedor
@app.callback(
    [Output('predictor-table', 'data'),
     Output('predictor-table', 'columns'),
     Output('predictor-container', 'style')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_predictor_table(contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        columns = [{'name': i, 'id': i} for i in df.columns]
        data = df.to_dict('records')
        return data, columns, {'display': 'block'}
    return [], [], {'display': 'none'}

# Callback para mostrar la fila seleccionada en una tabla con el mismo formato
@app.callback(
    [Output('selected-row-table', 'data'),
     Output('selected-row-table', 'columns'),
     Output('selected-row-container', 'style'),
     Output('model-container', 'style')],
    Input('predictor-table', 'selected_rows'),
    State('predictor-table', 'data')
)
def display_selected_row(selected_rows, data):
    if selected_rows is None or len(selected_rows) == 0:
        return [], [], {'display': 'none'}, {'display': 'none'}
    selected_row = selected_rows[0]
    row_data = [data[selected_row]]  # Convertimos la fila seleccionada en una lista de un solo elemento
    columns = [{'name': i, 'id': i} for i in data[selected_row].keys()]
    return row_data, columns, {'display': 'block'}, {'display': 'block'}  # Mostramos la tabla y el contenedor de modelos

# Callback para mostrar el dropdown de clases cuando se selecciona una fila
@app.callback(
    [Output('class-selector', 'options'),
     Output('class-selector', 'value'),
     Output('class-selector', 'style'),
     Output('class-container', 'style')],
    Input('predictor-table', 'selected_rows'),
    State('predictor-table', 'data')
)
def update_class_selector(selected_rows, data):
    if selected_rows is None or len(selected_rows) == 0:
        return [], None, {'display': 'none'}, {'display': 'none'}
    
    selected_row = selected_rows[0]
    row_data = data[selected_row]

    # Mostrar el contenido del row_data para verificar la estructura
    print(f"Selected Data: {row_data}")

    # Asegurarnos de que existe la columna 'class'
    if 'class' not in row_data:
        return [], None, {'display': 'none'}, {'display': 'none'}

    # Obtenemos las posibles clases del dataset
    class_options = [{'label': i, 'value': i} for i in set([row['class'] for row in data])]
    current_class = row_data['class']
    
    # Mostramos tanto el dropdown como el contenedor
    return class_options, current_class, {'display': 'block'}, {'display': 'block'}


# Callback para ejecutar la lógica contrafactual y mostrar los resultados
@app.callback(
    [Output('results-table', 'data'),
     Output('results-table', 'columns'),
     Output('results-container', 'style')],
    Input('run-button', 'n_clicks'),
    State('predictor-table', 'selected_rows'),
    State('class-selector', 'value'),
    State('model-selector', 'value'),
    State('predictor-table', 'data')
)
def run_counterfactual(n_clicks, selected_rows, new_class, num_models, data):
    if n_clicks is None or len(selected_rows) == 0:
        return [], [], {'display': 'none'}

    # Obtener la fila seleccionada
    selected_row = selected_rows[0]
    row_data = data[selected_row]

    # Lógica para ejecutar el contrafactual (simplificada)
    original_row = row_data.copy()
    original_row['class'] = row_data['class']
    
    new_row = row_data.copy()
    new_row['class'] = new_class
    
    # Lógica de los modelos contrafactuales iría aquí
    
    results_data = [original_row, new_row]
    columns = [{'name': i, 'id': i} for i in row_data.keys()]
    
    return results_data, columns, {'display': 'block'}

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
