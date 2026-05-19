import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

dash.register_page(__name__, path='/diagnostics')
from GAStatus import shared_data_cache

layout = dbc.Container([
    dcc.Interval(id='ga-interval', interval=2000, n_intervals=0),

    dbc.Row([
        dbc.Col(html.H4("GA Diagnostics", className="text-secondary"), width=8),
        dbc.Col(dbc.Button("Back to Setup", href="/setup", color="secondary", outline=True), width=4,
                className="text-end")
    ], className="mb-4 mt-4"),

    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([html.Small("Progress"), html.H4(id='status-gen-readout', children="Iteration: 0 / 0")], width=3),
                dbc.Col([html.Small("Best Fitness"), html.H4(id='status-fitness-readout', className="text-success")],
                        width=2),
                dbc.Col([html.Small("Avg Diversity"), html.H4(id='status-div-readout', className="text-info")],
                        width=3),
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button("Pause", id='pause-btn', color="warning", outline=True),
                        dbc.Button("Stop GA", id='stop-btn', color="danger"),
                    ], className="w-100 mt-2")
                ], width=4),
            ], align="center")
        ])
    ], className="mb-4 shadow-sm border-0"),

    # this part is diagnostic layout
    # 1. Fitness evolution, 2. Standardize Selectn Diff, 3. Fitness Score vs Yield, 4. Drop down of scatter plots
    dbc.Row([
        # top left - 1. bottom left - 3.
        dbc.Col([
            dbc.Card([dbc.CardHeader("Fitness Evolution"), dbc.CardBody(dcc.Graph(id='graph-fit-gen'))],
                     className="mb-4"),
            dbc.Card([dbc.CardHeader("Fitness vs. Mean Yield"), dbc.CardBody(dcc.Graph(id='graph-fit-yield'))])
        ], width=6),

        # top right - 2. bottom right - 4
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Selection Diff vs. Diversity"),
                dbc.CardBody(dcc.Graph(id='graph-dg-div'))
            ], className="mb-4"),

            dbc.Card([
                dbc.Tabs([
                    dbc.Tab(label="Penalty Deductions", tab_id="tab-accounting"),
                    dbc.Tab(label="Population Plots", tab_id="tab-scatter"),
                ], id="analysis-tabs", active_tab="tab-accounting"),

                dbc.CardBody([
                    html.Div([
                        dbc.Row([
                            dbc.Col(html.Small("X-Axis:"), width=3),
                            dbc.Col(dcc.Dropdown(
                                id='secondary-trait-selector',
                                placeholder="Select Trait...",
                                style={'fontSize': '12px'}
                            ), width=9)
                        ], className="mb-2"),
                        dcc.Graph(id='graph-scatter-inspection')
                    ], id='container-scatter', style={'display': 'none'}),

                    # part of 4 - penalty stack plot
                    html.Div([
                        dcc.Graph(id='graph-penalty-bar')
                    ], id='container-accounting', style={'display': 'block'})
                ])
            ])
        ], width=6)
    ])
], fluid=True, className="p-4")


# help from gemini - I had issue getting the penalty stack plot to work with the scatter plot drop down
@dash.callback(
    [Output('container-scatter', 'style'),
     Output('container-accounting', 'style')],
    [Input('analysis-tabs', 'active_tab')]
)
def toggle_view(active_tab):
    if active_tab == "tab-scatter":
        return {'display': 'block'}, {'display': 'none'}

    return {'display': 'none'}, {'display': 'block'}


# data updater

@dash.callback(
    [Output('status-gen-readout', 'children'),
     Output('status-fitness-readout', 'children'),
     Output('status-div-readout', 'children'),
     Output('graph-fit-gen', 'figure'),
     Output('graph-fit-yield', 'figure'),
     Output('graph-dg-div', 'figure'),
     Output('graph-scatter-inspection', 'figure'),
     Output('graph-penalty-bar', 'figure'),
     Output('secondary-trait-selector', 'options')],
    [Input('ga-interval', 'n_intervals')],
    [State('secondary-trait-selector', 'value'),
     State('analysis-tabs', 'active_tab')],
    prevent_initial_call=True
)
def update_all_data(n, selected_trait, active_tab):
    if not shared_data_cache.latest_snapshot or shared_data_cache.optimizer is None:
        # I need 9 Outputs
        return [dash.no_update] * 9


    snap = shared_data_cache.latest_snapshot
    df_h = pd.DataFrame(shared_data_cache.history)
    pop_df = shared_data_cache.optimizer.breeding_pop.filtered_df
    primary = shared_data_cache.optimizer.primary_trait

    # Fitness convergence
    fig_fit = go.Figure()
    fig_fit.add_trace(go.Scatter(x=df_h['iteration'], y=df_h['best_fitness'],
                                 name="Max Fitness", line=dict(color="#2E7D32", width=3)))
    fig_fit.add_trace(go.Scatter(x=df_h['iteration'], y=df_h['mean_fitness'],
                                 name="Mean Fitness", line=dict(color="#A5D6A7", dash='dot')))
    fig_fit.update_layout(template="simple_white", margin=dict(l=40, r=20, t=20, b=40),
                          hovermode="x unified",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # Fitness vs Mean Yield/primary trait
    fig_yield = go.Figure()
    fig_yield.add_trace(go.Scatter(x=df_h['mean_yield'], y=df_h['best_fitness'],
                                   mode='lines+markers', name="Trend",
                                   line=dict(color="#1976D2", width=2),
                                   marker=dict(size=6, symbol="diamond")))
    fig_yield.update_layout(xaxis_title=f"Mean {primary.upper()}", yaxis_title="Max Fitness",
                            template="simple_white", margin=dict(l=40, r=20, t=20, b=40))

    # Selection diff vs diversity
    fig_dg = go.Figure()

    if not df_h.empty:
        # plot the historical path - all points except the last one
        history_df = df_h.iloc[:-1] if len(df_h) > 1 else df_h

        fig_dg.add_trace(go.Scatter(
            x=history_df['diversity'],
            y=history_df['delta_g'],
            mode='lines+markers',
            name="Path",
            line=dict(color="#FF7043", width=2),
            marker=dict(size=6, opacity=0.7)
        ))

        # add trace for the current one
        final_point = df_h.iloc[[-1]]

        fig_dg.add_trace(go.Scatter(
            x=final_point['diversity'],
            y=final_point['delta_g'],
            mode='markers',
            name="Current Best",
            marker=dict(
                size=12,
                color="#008cff",
                line=dict(width=2, color="#FF7043")
            )
        ))


    fig_dg.update_layout(
        xaxis_title="Diversity (1-COA)",
        yaxis_title="Standardized Selection Diff",
        template="simple_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # scatter plot in dropdown
    fig_scat = go.Figure()
    if selected_trait and selected_trait in pop_df.columns:
        fig_scat.add_trace(go.Scatter(x=pop_df[primary], y=pop_df[selected_trait],
                                      mode='markers', name="Population",
                                      marker=dict(color='rgba(200, 200, 200, 0.3)', size=6)))
        sel_df = pop_df.loc[snap['selected_ids']]
        fig_scat.add_trace(go.Scatter(x=sel_df[primary], y=sel_df[selected_trait],
                                      mode='markers', name="Selected",
                                      marker=dict(color='rgba(46, 125, 50, 0.8)', size=8, line=dict(width=1, color='white'))))
        fig_scat.update_layout(xaxis_title=primary.upper(), yaxis_title=selected_trait.upper())
    fig_scat.update_layout(template="simple_white", margin=dict(l=40, r=20, t=20, b=40))


    # stacked penalty plot
    fig_bar = go.Figure()

    if shared_data_cache.optimizer and shared_data_cache.ga_instance and snap:
        opt = shared_data_cache.optimizer
        ga_instance = shared_data_cache.ga_instance

        population = ga_instance.population
        fitness_scores = ga_instance.last_generation_fitness

        try:
            # calc raw yields for everyone in the current population
            pop_yields = [opt.breeding_pop.filtered_df.iloc[sol][opt.primary_trait].mean() for sol in population]
            best_idx = np.argmax(fitness_scores)
            greedy_idx = np.argmax(pop_yields)
        except Exception:
            return [dash.no_update] * 9

        comparison_indices = [best_idx, greedy_idx]
        labels = ['Chosen (Best Fitness)', 'Greedy (Max Yield)']
        traces_data = {}

        for i, idx in enumerate(comparison_indices):
            sol = population[idx]
            df_sol = opt.breeding_pop.filtered_df.iloc[sol]

            # a. base yield
            traces_data.setdefault('Base Yield', [0, 0])[i] = df_sol['norm_'+opt.primary_trait].mean()

            # b. trait penalties
            for t in opt.sec_trait_list:
                t_label = f'{t.name.upper()} Pen'
                pen_val = -np.sum(t.penalty_logic(df_sol['norm_'+t.name].values))
                traces_data.setdefault(t_label, [0, 0])[i] = pen_val

            # c. parent overuse and div bonus
            parents = pd.concat([df_sol['parent1'], df_sol['parent2']])
            counts = parents.value_counts()

            valid_pars = [p for p in counts.index if p in opt.breeding_pop.coa_matrix.index]

            total_slots = len(parents)
            p_vec = np.array([counts[p] / total_slots for p in valid_pars])

            rel_mat = opt.breeding_pop.coa_matrix.loc[valid_pars, valid_pars].values

            group_kinship = p_vec.T @ rel_mat @ p_vec

            mask = np.eye(rel_mat.shape[0], dtype=bool)
            max_pairwise = np.max(rel_mat[~mask]) if rel_mat.size > 1 else 0
            e = 1 if max_pairwise >= opt.div_target else 0

            counts = parents.value_counts()
            max_count = counts.max()
            excess_use = ((max_count - opt.max_parent_use) / opt.max_parent_use)**2

            # excess_use = 1 if counts.max() > self.max_parent_use else 0
            div_trig = e + excess_use

            # diversity_penalty = group_kinship * e * excess_use * opt.div_penalty
            diversity_penalty = group_kinship * div_trig * opt.div_penalty

            # linear scaling for plotting
            div_pen = (diversity_penalty/2) if diversity_penalty > 0 else diversity_penalty


            # # parents = pd.concat([df_sol['parent1'], df_sol['parent2']])
            # counts = parents.value_counts()
            # excess = counts[counts > opt.max_parent_use] - opt.max_parent_use
            # traces_data.setdefault('Parent Penalty', [0, 0])[i] = -np.sum((excess ** 2) * opt.parent_overuse_penalty)
            #
            # u_p = parents.unique()
            # sub_coa = opt.breeding_pop.coa_matrix.loc[u_p, u_p]
            # avg_coa = (sub_coa.values.sum() - len(u_p)) / (len(u_p) ** 2 - len(u_p))
            # d_impact = opt.div_bonus if (1 - avg_coa) >= opt.div_target else -opt.div_penalty

            traces_data.setdefault('Diversity Impact', [0, 0])[i] = div_pen


        colors_seq = px.colors.qualitative.T10

        for i, (name, values) in enumerate(traces_data.items()):

            if name == 'Base Yield':
                color = '#8db48e'
            elif name == 'Diversity Impact':
                color = '#4b8b9c'
            elif 'HT' in name:
                color = '#a05252'
            else:
                color = colors_seq[i % len(colors_seq)]

            fig_bar.add_trace(go.Bar(
                name=name, x=labels, y=values,
                marker_color=color,
                marker_line=dict(width=0.5, color='white')
            ))

        # add net fitness labels at the top of the stack
        for i, idx in enumerate(comparison_indices):
            # finds top of stack
            top_y = traces_data['Base Yield'][i] + sum(
                [v[i] for k, v in traces_data.items() if v[i] > 0 and k != 'Base Yield'])
            fig_bar.add_annotation(
                x=labels[i], y=top_y + 1,
                text=f"Net: {fitness_scores[idx]:.1f}",
                showarrow=False, font=dict(color="white", size=14, family="Arial Black"),
                # bgcolor="rgba(0,0,0,0.7)",
                # borderpad=6
            )

    fig_bar.update_layout(
        barmode='relative',  # creates the downward stacking
        template="simple_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,30,1)',
        yaxis_title="Fitness",
        title="Stacked Penalties",
        margin=dict(l=60, r=20, t=80, b=40)
    )


    trait_options = [
        {'label': str(c).upper(), 'value': c}
        for c in pop_df.columns if pd.api.types.is_numeric_dtype(pop_df[c])
    ]

    return (
        f"Iteration: {snap['iteration']} / {snap['max_gen']}",  # 1. status-gen-readout
        f"{snap['best_fitness']:.2f}",  # 2. status-fitness-readout
        f"{snap['diversity']:.3f}",  # 3. status-div-readout
        fig_fit,  # 4. graph-fit-gen
        fig_yield,  # 5. graph-fit-yield
        fig_dg,  # 6. graph-dg-div
        fig_scat,  # 7. graph-scatter-inspection
        fig_bar,  # 8. graph-penalty-bar
        trait_options  # 9. secondary-trait-selector.options
    )
