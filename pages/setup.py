import dash
from dash import dcc, html, Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64
import io
import threading
import pygad
dash.register_page(__name__, path='/setup')

from GAStatus import shared_data_cache
from GeneticAlgorithm import GeneticAlgorithm
from BreedingPopulation import BreedingPopulation
from TraitRules import TraitRules





def create_trait_card(trait_name, min_val, max_val):
    return dbc.Col([
        dbc.Card([
            dbc.CardHeader(html.B(trait_name.upper())),
            dbc.CardBody([
                dcc.Graph(
                    id={'type': 'trait-hist', 'index': trait_name},
                    config={'displayModeBar': False},
                    style={'height': '200px'}
                ),
                dbc.Row([
                    dbc.Col([
                        html.Label("L-Prune", style={'fontSize': '10px'}),
                        dbc.Input(id={'type': 'l-prune', 'index': trait_name}, type='number', value=None, placeholder=f"{min_val:.1f}", size="sm"),
                    ], width=3),
                    dbc.Col([
                        html.Label("L-Thresh", style={'fontSize': '10px'}),
                        dbc.Input(id={'type': 'l-thresh', 'index': trait_name}, type='number', value=min_val,
                                  size="sm"),
                    ], width=3),
                    dbc.Col([
                        html.Label("U-Thresh", style={'fontSize': '10px'}),
                        dbc.Input(id={'type': 'u-thresh', 'index': trait_name}, type='number', value=max_val,
                                  size="sm"),
                    ], width=3),
                    dbc.Col([
                        html.Label("U-Prune", style={'fontSize': '10px'}),
                        dbc.Input(id={'type': 'u-prune', 'index': trait_name}, type='number', value=max_val, size="sm"),
                    ], width=3),
                ], className="g-1"),
                html.Div([
                    html.Label("Weight", style={'fontSize': '10px'}),
                    dcc.Slider(0, 10, 0.5, value=3, id={'type': 'weight', 'index': trait_name}),
                ], className="mt-2")
            ])
        ], className="shadow-sm mb-3")
    ], width=3)  # I changed to 3 to fit 4 cards per row - might need to reconsider to be dynamic


def get_upload_component(id_label, title):
    return dcc.Upload(
        id=id_label,
        children=html.Div([title, html.A(' Select File')]),
        style={
            'width': '100%', 'height': '45px', 'lineHeight': '45px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center'
        }
    )


# APP LAYOUT

layout = dbc.Container([
    dcc.Store(id='raw-data-store'),

    dbc.Card([
        dbc.CardHeader(html.H4("Optimization Configuration", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                # upload files
                dbc.Col([
                    get_upload_component('upload-pheno', '1. Phenotype'),
                    get_upload_component('upload-coa', '2. COA Matrix'),
                ], width=3),

                # GA params
                dbc.Col([
                    html.Label("Primary Trait", className="small"),
                    dcc.Dropdown(id='primary-trait-dropdown'),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select N", className="small"),
                            dbc.Input(id='num-select', type='number', value=100, size="sm"),
                        ]),
                        dbc.Col([
                            html.Label("Max Parent", className="small"),
                            dbc.Input(id='max-parent', type='number', value=5, size="sm"),
                        ]),
                    ], className="mt-2")
                ], width=3),

                # this is for discard stats - updates live as pruning values are set
                # also has the run button that move to diagnostics
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Small("Lines Remaining:"),
                            html.H4(id='live-counter', children="N/A", className="text-primary mb-0")
                        ], className="p-2")
                    ], className="mb-2 text-center"),
                    dbc.Button("Run Optimizer", href='/diagnostics', id='init-btn', color="primary", className="w-100", size="lg")
                ], width=3),


                dbc.Col([
                    html.Label("Diversity Target (1 - COA)", className="small font-weight-bold"),
                    dcc.Slider(
                        id='diversity-target-slider',
                        min=0, max=1, step=0.05,
                        value=0.25,  # default target
                        marks={0: '0', 0.5: '0.5', 1: '1'},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    html.Div([
                        html.Small("Avg Div (Original Pop): "),
                        html.Span(id='avg-diversity-readout', children="N/A", className="text-info font-weight-bold")
                    ], className="mt-2 text-center")
                ], width=2),

                # data upload status
                dbc.Col([
                    html.Div(id='upload-status', className="small text-muted text-center pt-2")
                ], width=1)
            ])
        ])
    ], className="mb-4 shadow-sm"),

    html.H4("Population Filtering", className="mb-4"),
    dbc.Row(id='trait-grid-container', children=[
        html.Div("Awaiting data upload to generate histograms...", className="p-5 text-center text-muted")
    ])
], fluid=True, className="p-4")




@dash.callback(
    [Output('raw-data-store', 'data'),
     Output('trait-grid-container', 'children'),
     Output('primary-trait-dropdown', 'options'),
     Output('upload-status', 'children'),
     Output('avg-diversity-readout', 'children')],
    [Input('upload-pheno', 'contents'),
     Input('upload-coa', 'contents')],
    [State('upload-pheno', 'filename'),
     State('upload-coa', 'filename')],
    prevent_initial_call=True
)
def handle_uploads(pheno_contents, coa_contents, pheno_name, coa_name):

    if pheno_contents is None:
        return dash.no_update, dash.no_update, dash.no_update, "Waiting for Phenotype...", "N/A"

    # process pheno data
    content_type, content_string = pheno_contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    # janitor for column names
    df.columns = df.columns.str.lower()

    avg_div_text = "N/A"
    if coa_contents:
        _, coa_string = coa_contents.split(',')
        decoded_coa = base64.b64decode(coa_string)

        try:
            # read pickle file but fails gracefully
            try:
                coa_df = pd.read_pickle(io.BytesIO(decoded_coa))
            except:
                coa_df = pd.read_csv(io.StringIO(decoded_coa.decode('utf-8')), index_col=0)

            shared_data_cache.coa_matrix = coa_df

            # initial population diversity: 1 - mean(COA)
            avg_div_val = 1 - coa_df.values.mean()
            avg_div_text = f"{avg_div_val:.3f}"

        except Exception as e:
            return dash.no_update, dash.no_update, dash.no_update, f"Error loading COA: {str(e)}", "N/A"

    # Card UI for traits
    # takes out possible non trait names
    exclude = ['line', 'pedigree', 'parent1', 'parent2', 'cross_id', 'id', 'parent_1', 'parent_2']
    traits = [col for col in df.columns if col not in exclude and pd.api.types.is_numeric_dtype(df[col])]

    # card for traits
    cards = [create_trait_card(t, float(df[t].min()), float(df[t].max())) for t in traits]

    grid = [dbc.Row(cards[i:i + 4], className="mb-4") for i in range(0, len(cards), 4)]

    options = [{'label': t.upper(), 'value': t} for t in traits]

    status_msg = html.Div([
        html.P(f"✅ Phenotype: {pheno_name}", className="mb-1 text-success"),
        html.P(f"{'✅ COA Matrix: ' + coa_name if coa_contents else '❌ COA Matrix Missing'}",
               className="mb-0",
               style={'color': 'green' if coa_contents else 'red'})
    ])

    return df.to_dict('records'), grid, options, status_msg, avg_div_text


@dash.callback(
    Output({'type': 'trait-hist', 'index': MATCH}, 'figure'),
    [Input('raw-data-store', 'data'),
     Input({'type': 'l-prune', 'index': MATCH}, 'value'),
     Input({'type': 'u-prune', 'index': MATCH}, 'value'),
     Input({'type': 'l-thresh', 'index': MATCH}, 'value'),
     Input({'type': 'u-thresh', 'index': MATCH}, 'value')],
    State({'type': 'trait-hist', 'index': MATCH}, 'id'),
    prevent_initial_call=True
)
def update_histogram(data, lp, up, lt, ut, hist_id):
    if not data: return go.Figure()

    trait_name = hist_id['index']
    df = pd.DataFrame(data)
    vals = df[trait_name]
    d_min, d_max = vals.min(), vals.max()


    lp = lp if lp is not None else d_min
    up = up if up is not None else d_max
    lt = lt if lt is not None else lp  # Penalty starts at prune if no thresh
    ut = ut if ut is not None else up  # Penalty starts at prune if no thresh

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=vals, marker_color='#2E7D32', opacity=0.7))

    # add visual boundaries for pruning and thresholding
    fig.add_vrect(x0=d_min, x1=lp, fillcolor="red", opacity=0.2, line_width=0)
    fig.add_vrect(x0=up, x1=d_max, fillcolor="red", opacity=0.2, line_width=0)
    fig.add_vrect(x0=lp, x1=lt, fillcolor="orange", opacity=0.2, line_width=0)
    fig.add_vrect(x0=ut, x1=up, fillcolor="orange", opacity=0.2, line_width=0)

    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), template='simple_white', height=200)
    return fig


@dash.callback(
    Output('live-counter', 'children'),
    [Input('raw-data-store', 'data'),
     Input({'type': 'l-prune', 'index': ALL}, 'value'),
     Input({'type': 'u-prune', 'index': ALL}, 'value')],
    State({'type': 'l-prune', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def update_global_counter(data, l_prunes, u_prunes, ids):
    if not data: return "N/A"
    df = pd.DataFrame(data)
    total = len(df)

    for i in range(len(ids)):
        trait = ids[i]['index']
        low = l_prunes[i] if l_prunes[i] is not None else -np.inf
        high = u_prunes[i] if u_prunes[i] is not None else np.inf
        df = df[(df[trait] >= low) & (df[trait] <= high)]

    return f"{len(df):,} / {total:,}"


@dash.callback(
    Output('url', 'pathname', allow_duplicate=True),
    Input('init-btn', 'n_clicks'),
    [State('raw-data-store', 'data'),
     State('primary-trait-dropdown', 'value'),
     State('num-select', 'value'),
     State('max-parent', 'value'),
     State('diversity-target-slider', 'value'),
     State({'type': 'l-prune', 'index': ALL}, 'value'),
     State({'type': 'u-prune', 'index': ALL}, 'value'),
     State({'type': 'l-thresh', 'index': ALL}, 'value'),
     State({'type': 'u-thresh', 'index': ALL}, 'value'),
     State({'type': 'weight', 'index': ALL}, 'value'),
     State({'type': 'l-prune', 'index': ALL}, 'id')],
    prevent_initial_call=True
)
def launch_optimizer(n_clicks, data, primary_trait, n_select, max_p, div_target,
                     l_prunes, u_prunes, l_threshs, u_threshs, weights, ids):

    if not n_clicks or not data or shared_data_cache.coa_matrix is None:
        return dash.no_update

    df = pd.DataFrame(data)

    sec_traits = []
    for i in range(len(ids)):
        trait_name = ids[i]['index']
        trait_std = df[trait_name].std()

        # If std is 0 (all values same), avoid division by zero
        if trait_std == 0: trait_std = 1.0

        normalized_weight = weights[i] / trait_std

        is_primary = (trait_name == primary_trait)

        rule = TraitRules(
            name=trait_name,
            lower_prune=l_prunes[i],
            upper_prune=u_prunes[i],
            # lower_thresh=l_threshs[i],
            # upper_thresh=u_threshs[i],
            # penalty_weight=normalized_weight
            lower_thresh=None if is_primary else l_threshs[i],
            upper_thresh=None if is_primary else u_threshs[i],
            penalty_weight=0.0 if is_primary else normalized_weight
        )
        sec_traits.append(rule)


    # make an instance of BreedingPop and do pruning immediately
    # so that only filtered_df is used in GA
    wheat_pop = BreedingPopulation(df, shared_data_cache.coa_matrix)
    wheat_pop.do_pruning(sec_traits)

    if len(wheat_pop.filtered_df) < n_select:
        # TODO: update to give signal in UI
        return dash.no_update


    optimizer = GeneticAlgorithm(
        breeding_pop=wheat_pop,
        primary_trait=primary_trait,
        sec_trait_list=sec_traits,
        selection_size=n_select,
        max_parent_use=max_p,
        div_target=div_target
    )

    def on_gen(ga_instance):
        # TODO: Define stop and pause GA actions for buttons
        if optimizer.stop_signal:
            return "stop"

        # gemini help to create snapshot and push to shared memory - helps with page nav
        snapshot = optimizer.create_snapshot(ga_instance)
        shared_data_cache.latest_snapshot = snapshot
        shared_data_cache.history.append(snapshot)

        print(f"Iteration {ga_instance.generations_completed} complete. Best: {snapshot['best_fitness']:.2f}")

    # 8. Set up PyGAD
    ga_instance = pygad.GA(
        num_generations=100,
        num_parents_mating=15,
        fitness_func=optimizer.fitness_function,
        sol_per_pop=50,
        num_genes=n_select,
        gene_space=list(range(len(wheat_pop.filtered_df))),
        parent_selection_type='sss',
        keep_parents=2,
        crossover_type='single_point',
        mutation_type='random',
        mutation_probability=0.1,
        allow_duplicate_genes=False,
        on_generation=on_gen
    )

    shared_data_cache.active_ga = ga_instance
    shared_data_cache.optimizer = optimizer
    shared_data_cache.history = []


    def run_ga_logic():
        shared_data_cache.ga_instance = ga_instance

        ga_instance.run()

        # this is the last generation
        shared_data_cache.latest_snapshot = optimizer.create_snapshot(ga_instance)

    ga_thread = threading.Thread(target=run_ga_logic, daemon=True)
    ga_thread.start()

    return "/diagnostics"

#

@dash.callback(
    [
        Output({'type': 'l-thresh', 'index': ALL}, 'disabled'),
        Output({'type': 'u-thresh', 'index': ALL}, 'disabled'),
        Output({'type': 'weight', 'index': ALL}, 'disabled'),
        Output({'type': 'l-thresh', 'index': ALL}, 'value'),
        Output({'type': 'u-thresh', 'index': ALL}, 'value'),
        Output({'type': 'weight', 'index': ALL}, 'value')  # Add this Output
    ],
    [Input('primary-trait-dropdown', 'value')],
    [State({'type': 'l-thresh', 'index': ALL}, 'id')],
    prevent_initial_call=True
)
def disable_primary_trait_inputs(primary_trait, ids):
    if not primary_trait or not ids:
        return [dash.no_update] * 6

    # 1. Which ones to disable
    disabled_states = [id_dict['index'] == primary_trait for id_dict in ids]

    # 2. Thresholds: Clear them (None) if primary
    thresh_values = [None if id_dict['index'] == primary_trait else dash.no_update for id_dict in ids]

    # 3. Weights: Set to 0 if primary, otherwise leave alone
    weight_values = [0 if id_dict['index'] == primary_trait else dash.no_update for id_dict in ids]

    return (
        disabled_states, # l-thresh disabled
        disabled_states, # u-thresh disabled
        disabled_states, # weight disabled
        thresh_values,   # l-thresh values
        thresh_values,   # u-thresh values
        weight_values    # weight values (Now shows 0 for primary)
    )
