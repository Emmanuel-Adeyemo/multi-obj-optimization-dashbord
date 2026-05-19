import dash
from dash import dcc, html, Input, Output, dash_table, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
import numpy as np


dash.register_page(__name__, path='/summary')
from GAStatus import shared_data_cache



def layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H4("Selection Summary", className="text-secondary"), width=8),
            dbc.Col(dbc.Button("Back to Diagnostics", href="/diagnostics", color="secondary", outline=True), width=4,
                    className="text-end")
        ], className="mb-4 mt-4"),

        # row 1 - top level stats
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6(f"Standardized Selection Diff.", className="card-subtitle text-muted"),
                    html.H3(id='summary-gain-kpi', className="text-success")
                ])
            ]), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Avg Diversity (Selected Pop)", className="card-subtitle text-muted"),
                    html.H3(id='summary-div-kpi', className="text-info")
                ])
            ]), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Selection Overlap (Chosen vs Greedy)", className="card-subtitle text-muted"),
                    html.H3(id='summary-overlap-kpi', className="text-warning")
                ])
            ]), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Unique Parents Used", className="card-subtitle text-muted"),
                    html.H3(id='summary-parent-kpi', className="text-primary")
                ])
            ]), width=3),
        ], className="mb-4"),

        # row 2 - plots
        dbc.Row([
            # density plots
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Row([
                            dbc.Col(html.B("Trait Distribution Shift"), width=6),
                            dbc.Col(dcc.Dropdown(id='summary-trait-selector', placeholder="Select Trait..."),
                                    width=6, style={"fontSize": "12px"})
                        ])
                    ]),
                    dbc.CardBody(dcc.Graph(id='graph-distribution-shift'))
                ])
            ], width=7),

            # RIGHT: Parental Treemap
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.B("Parental Contribution (Pedigree)")),
                    dbc.CardBody(dcc.Graph(id='graph-parent-treemap'))
                ])
            ], width=5),
        ], className="mb-4"),

        # row 3 - datatable
        dbc.Card([
            dbc.CardHeader(dbc.Row([
                dbc.Col(html.B("Selected Population"), width=6),
                dbc.Col(dbc.Button("Download CSV", id="btn-csv", color="light", size="sm"), width=6,
                        className="text-end")
            ])),
            dbc.CardBody([
                dash_table.DataTable(
                    id='summary-table',
                    page_size=15,
                    sort_action="native",
                    filter_action="native",
                    export_format="csv",
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'fontSize': '12px'},
                    style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                    style_data_conditional=[{
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }]
                ),
                dcc.Download(id="download-dataframe-csv"),
            ])
        ], className="mb-5")
    ], fluid=True)




@callback(
    [Output('summary-gain-kpi', 'children'),
     Output('summary-div-kpi', 'children'),
     Output('summary-overlap-kpi', 'children'),
     Output('summary-parent-kpi', 'children'),
     Output('summary-trait-selector', 'options'),
     Output('summary-trait-selector', 'value'),
     Output('graph-parent-treemap', 'figure'),
     Output('summary-table', 'data'),
     Output('summary-table', 'columns')],
    [Input('summary-gain-kpi', 'id')]  # Triggered on load
)
def populate_summary(_):
    if not shared_data_cache.latest_snapshot or not shared_data_cache.optimizer or not shared_data_cache.active_ga:
        return ["N/A"] * 4 + [[], None, go.Figure(), [], []]

    opt = shared_data_cache.optimizer
    snap = shared_data_cache.latest_snapshot
    ga_instance = shared_data_cache.active_ga

    # selection overlap logic - selected by fitness vs greedy

    population = ga_instance.population
    fitness_scores = ga_instance.last_generation_fitness
    best_idx = np.argmax(fitness_scores)

    p_trait = [opt.breeding_pop.filtered_df.iloc[solution][opt.primary_trait].mean() for solution in population]
    greedy_idx = np.argmax(p_trait)

    best_sol = population[best_idx]
    selected_df = opt.breeding_pop.filtered_df.iloc[best_sol]

    selected_ids = selected_df.nlargest(opt.selection_size, opt.primary_trait).index

    greedy_sol = population[greedy_idx]
    greedy_df = opt.breeding_pop.filtered_df.iloc[greedy_sol]
    greedy_top_n = greedy_df.nlargest(opt.selection_size, opt.primary_trait).index

    # get selected data from snapshot
    # selected_ids = snap['selected_ids']
    full_df = opt.breeding_pop.df
    # selected_df = opt.breeding_pop.filtered_df.loc[selected_ids]


    # Get top N by primary trait alone
    # greedy_top_n = opt.breeding_pop.filtered_df.nlargest(opt.selection_size, opt.primary_trait).index
    overlap_count = len(set(selected_ids).intersection(set(greedy_top_n)))
    overlap_pct = f"{(overlap_count / opt.selection_size) * 100:.0f}%"

    # gain
    gain_text = f"{opt.primary_trait.title()}: +{snap['delta_g']:.2f} σ"
    div_text = f"{snap['diversity']:.3f}"

    parents_series = pd.concat([selected_df['parent1'], selected_df['parent2']])
    parent_counts = parents_series.value_counts().reset_index()
    parent_counts.columns = ['Parent', 'Count']
    unique_parents = f"{len(parent_counts)}"

    # tree map for parent use
    fig_tree = px.treemap(parent_counts, path=['Parent'], values='Count',
                          color='Count', color_continuous_scale='Blues')
    fig_tree.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    # dropdown for density
    numeric_cols = [c for c in full_df.columns if pd.api.types.is_numeric_dtype(full_df[c])]
    options = [{'label': c.upper(), 'value': c} for c in numeric_cols]

    # selection data table
    table_data = selected_df.to_dict('records')
    table_cols = [{"name": i.upper(), "id": i} for i in selected_df.columns]

    return (gain_text, div_text, overlap_pct, unique_parents, options, opt.primary_trait,
            fig_tree, table_data, table_cols)




@callback(
    Output('graph-distribution-shift', 'figure'),
    [Input('summary-trait-selector', 'value')]
)
def update_shift_plot(selected_trait):
    if not selected_trait or not shared_data_cache.optimizer:
        return go.Figure()

    opt = shared_data_cache.optimizer
    snap = shared_data_cache.latest_snapshot

    # data for plotting
    original_vals = opt.breeding_pop.df[selected_trait].dropna().tolist()
    selected_vals = opt.breeding_pop.filtered_df.loc[snap['selected_ids'], selected_trait].dropna().tolist()

    hist_data = [original_vals, selected_vals]
    group_labels = ['Overall Population', 'Selected Group']
    # colors = ['#bdc3c7', '#2E7D32']

    fig = ff.create_distplot(
        hist_data,
        group_labels,
        colors=['#bdc3c7', 'rgba(46, 125, 50, 0.1)'],
        show_hist=False,
        show_rug=False,
        bin_size=5
    )

    # fill density plot
    fig.update_traces(fill='tozeroy', opacity=0.2, selector=dict(type='scatter'))

    # add mean lines
    orig_mean = np.mean(original_vals)
    sel_mean = np.mean(selected_vals)

    # original data line
    fig.add_vline(
        x=orig_mean,
        line_dash="dash",
        line_color="#34495e",
        line_width=2,
        layer='above',
        annotation_text=f"Global: {orig_mean:.2f}",
        annotation_position="top left"
    )

    # selected data mean line
    fig.add_vline(
        x=sel_mean,
        line_color="#1b5e20",
        line_width=3,
        layer='above',
        annotation_text=f"Selected: {sel_mean:.2f}",
        annotation_position="top right"
    )


    fig.update_layout(
        template="simple_white",
        title=f"<b>Distribution Shift: {selected_trait.upper()}</b>",
        xaxis_title=selected_trait.lower(),
        yaxis_title="Density",
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig
