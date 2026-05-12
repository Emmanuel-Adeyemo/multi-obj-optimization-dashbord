import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc


app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True
)

# NAVBAR
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("1. Setup", href="/setup")),
        dbc.NavItem(dbc.NavLink("2. Diagnostics", href="/diagnostics")),
        dbc.NavItem(dbc.NavLink("3. Summary", href="/summary")),
    ],
    brand="Selection Optimizer",
    brand_href="/setup",
    color="primary",
    dark=True,
    className="mb-4 shadow-sm"
)

# MAIN
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,

    dbc.Container(dash.page_container, fluid=True)
])

if __name__ == '__main__':
    app.run(debug=True, port=8050)
