"""Plant setup and simulation viewer.

python -m F00_Dash.app            then open http://127.0.0.1:8050

Left: plant setup (finishing stations, stage 1/2 machines, operators, buffers, products), saved as JSON
in D00_Plant/configs/. Right: run one simulated episode with a policy and inspect KPIs and Gantt charts
(finishing stations, machines, operators) plus buffer levels. Prep settings are kept from the loaded file.
"""
import glob
import os

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dash_table, dcc, html, Input, Output, State, ctx, no_update

from D00_Plant import PlantConfig, PlantEnv, default_config
from D00_Plant.config import ProductCfg
from D00_Plant.policies import POLICIES

CONFIG_DIR = 'D00_Plant/configs'
EVENTS_PATH = 'runs/dash/events.csv'

PRODUCT_COLORS = ['#2a6fdb', '#e8871e', '#2e9e5b', '#c23b5a', '#7b5cc4', '#1aa3a3', '#9c7a2e', '#5d6d7e']
STATE_COLORS = {
    'idle': '#d9dde3', 'wait_op': '#b8c4d6', 'changeover': '#f2c14e', 'starved_comp': '#f08c6c',
    'starved_input': '#e05252', 'blocked': '#8e44ad', 'broken': '#333333',
    'travel': '#f2c14e', 'work S1': '#2a6fdb', 'work S2': '#2e9e5b',
}
STATE_PATTERNS = {'starved_comp': '/', 'starved_input': 'x', 'blocked': '\\'}   # problem states are hatched
PRODUCT_COLUMNS = [
    {'name': 'Product', 'id': 'name'},
    {'name': 'Stage 1 (min/unit)', 'id': 'stage1_time', 'type': 'numeric'},
    {'name': 'Stage 2 (min/unit)', 'id': 'stage2_time', 'type': 'numeric'},
    {'name': 'Finishing (min/cycle)', 'id': 'finish_time', 'type': 'numeric'},
    {'name': 'Stations share (%)', 'id': 'share', 'type': 'numeric'},
]

# Form fields: (component id, config path, label)
FIELDS = [
    ('Finishing', [('fin-stations', ('finishing', 'n_stations'), 'Stations'),
                   ('fin-slots', ('finishing', 'slots'), 'Units per cycle'),
                   ('fin-ops', ('finishing', 'n_operators'), 'Finishing operators'),
                   ('fin-load', ('finishing', 'load_time'), 'Load time (min)')]),
    ('Stage 1', [('s1-n', ('stage1', 'n_machines'), 'Machines'),
                 ('s1-co', ('stage1', 'changeover_time'), 'Changeover (min)')]),
    ('Stage 2', [('s2-n', ('stage2', 'n_machines'), 'Machines'),
                 ('s2-co', ('stage2', 'changeover_time'), 'Changeover (min)')]),
    ('Operators', [('op-n', ('operators', 'n_operators'), 'Building operators'),
                   ('op-travel', ('operators', 'travel_time'), 'Travel time (min)'),
                   ('op-patience', ('operators', 'patience'), 'Patience (min)')]),
    ('Buffers', [('buf-1', ('buffers', 'stage1_capacity'), 'Stage-1 buffer capacity'),
                 ('buf-2', ('buffers', 'stage2_capacity'), 'Stage-2 buffer capacity')]),
]
FIELD_LIST = [f for _, group in FIELDS for f in group]
INT_FIELDS = {'n_stations', 'slots', 'n_operators', 'n_machines', 'stage1_capacity', 'stage2_capacity'}

CARD = {'background': 'white', 'border': '1px solid #e1e4e8', 'borderRadius': '8px', 'padding': '12px 16px',
        'marginBottom': '12px'}
LABEL = {'fontSize': '13px', 'color': '#555', 'display': 'block', 'marginBottom': '2px'}
INPUT = {'width': '100%', 'padding': '4px 6px', 'boxSizing': 'border-box'}


# ---------------------------------------------------------------------- config <-> form
def config_files():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(CONFIG_DIR, '*.json')))


def policy_options():
    opts = [{'label': f'{name} (baseline)', 'value': name} for name in POLICIES]
    for path in sorted(glob.glob('runs/*/best.pt')):
        opts.append({'label': f'trained: {path.split(os.sep)[1]}', 'value': path})
    return opts


def config_to_form(cfg):
    d = cfg.to_dict()
    values = [d[section][key] for _, (section, key), _ in FIELD_LIST]
    rows = [{'name': p.name, 'stage1_time': p.stage1_time, 'stage2_time': p.stage2_time,
             'finish_time': p.finish_time, 'share': round(100 * p.finish_share, 2)} for p in cfg.products]
    return values, rows


def form_to_config(base, values, rows):
    """Apply form values onto a base config dict; returns (PlantConfig, list of warnings)."""
    d = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for (_, (section, key), label), v in zip(FIELD_LIST, values):
        if v is None or v < 0:
            raise ValueError(f'{label}: enter a positive number')
        d[section][key] = int(v) if key in INT_FIELDS else float(v)
    rows = [r for r in rows if r.get('name')]
    if not rows:
        raise ValueError('Add at least one product')
    if len({r['name'] for r in rows}) != len(rows):
        raise ValueError('Product names must be unique')
    for r in rows:
        for k in ('stage1_time', 'stage2_time', 'finish_time'):
            if not isinstance(r.get(k), (int, float)) or r[k] <= 0:
                raise ValueError(f"{r['name']}: times must be positive numbers")
    shares = np.array([float(r.get('share') or 0) for r in rows])
    if shares.sum() <= 0:
        raise ValueError('Station shares must add up to more than 0')
    warnings = [] if abs(shares.sum() - 100) < 1e-6 else [f'Station shares add up to {shares.sum():g}%, rescaled to 100%']
    d['products'] = [ProductCfg(r['name'], float(r['stage1_time']), float(r['stage2_time']), float(r['finish_time']),
                                float(s / shares.sum())).__dict__ for r, s in zip(rows, shares)]
    if len(rows) > d['max_products']:
        raise ValueError(f"At most {d['max_products']} products")
    if d['finishing']['n_stations'] < 1 or d['stage1']['n_machines'] < 1 or d['stage2']['n_machines'] < 1:
        raise ValueError('Need at least 1 station and 1 machine per stage')
    return PlantConfig.from_dict(d), warnings


def capacity_summary(cfg):
    """Rough steady-state capacities (units/h) to sanity-check a setup before simulating."""
    finish = cfg.nominal_finish_rate()
    w = np.array([p.finish_share for p in cfg.products])
    t1 = float(w @ [p.stage1_time for p in cfg.products])
    t2 = float(w @ [p.stage2_time for p in cfg.products])
    machines = min(cfg.stage1.n_machines * 60 / t1, cfg.stage2.n_machines * 60 / t2)
    operators = cfg.operators.n_operators * 60 / (t1 + t2)   # each unit needs t1 + t2 operator minutes
    building = min(machines, operators)
    limit = 'finishing' if finish <= building else ('operators' if operators < machines else 'machines')
    return (f'Finishing capacity {finish:.0f} units/h · building capacity {building:.0f} units/h '
            f'(machines {machines:.0f}, operators {operators:.0f}) · expected bottleneck: {limit}')


# ---------------------------------------------------------------------- simulation + figures
def simulate(cfg, hours, seed, policy_name):
    if policy_name in POLICIES:
        policy = POLICIES[policy_name]()
    else:
        from E00_MAPPO.model import load_policy   # torch only needed for trained policies
        policy, _ = load_policy(policy_name)
    env = PlantEnv(cfg, log_events=True, episode_hours=hours)
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        obs, _, terminated, truncated, info = env.step(policy(env, obs))
        done = terminated or truncated
    env.plant.close_timeline()
    return env, info['kpi']


def gantt(rows, entities, category, title, product_names):
    """rows: timeline tuples (entity, start, end, state, product, detail); one bar trace per category."""
    order = {e: i for i, e in enumerate(entities)}
    groups = {}
    for r in rows:
        if r[0] in order:
            groups.setdefault(category(r), []).append(r)
    fig = go.Figure()
    for cat in sorted(groups, key=lambda c: (not c.startswith('running'), c)):
        rs = groups[cat]
        if cat.startswith('running '):
            p = cat.split(' ', 1)[1]
            color = PRODUCT_COLORS[product_names.index(p) % len(PRODUCT_COLORS)] if p in product_names else '#888'
        else:
            color = STATE_COLORS.get(cat, '#999')
        fig.add_trace(go.Bar(
            name=cat, orientation='h', marker_color=color, marker_line_width=0,
            marker_pattern_shape=STATE_PATTERNS.get(cat, ''),
            y=[r[0] for r in rs], base=[r[1] / 60 for r in rs], x=[(r[2] - r[1]) / 60 for r in rs],
            customdata=[(r[3], r[4] or '-', r[5] or '-', r[1] / 60, r[2] / 60) for r in rs],
            hovertemplate='%{y}<br>%{customdata[0]} · %{customdata[1]}<br>%{customdata[2]}'
                          '<br>%{customdata[3]:.2f} h → %{customdata[4]:.2f} h<extra></extra>'))
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor='left', yref='container', y=1, yanchor='top', pad=dict(t=12)),
        barmode='overlay', bargap=0.15, template='plotly_white',
        height=max(300, 16 * len(entities) + 150), margin=dict(l=70, r=20, t=95, b=40),
        legend=dict(orientation='h', x=0, xanchor='left', y=1.0, yanchor='bottom', font_size=11),
        xaxis_title='hours')
    fig.update_yaxes(categoryorder='array', categoryarray=entities[::-1], tickfont_size=10)
    return fig


def empty_figure(text='Run a simulation to see this chart'):
    fig = go.Figure()
    fig.add_annotation(text=text, showarrow=False, font=dict(size=15, color='#888'))
    fig.update_layout(template='plotly_white', height=300, xaxis_visible=False, yaxis_visible=False)
    return fig


def machine_category(r):
    return f'running {r[4]}' if r[3] == 'running' else r[3]


def operator_category(r):
    return f"work {r[5][:2]}" if r[3] == 'work' else r[3]


def buffer_figure(plant, cfg):
    fig = go.Figure()
    if plant.samples:
        t = [s[0] / 60 for s in plant.samples]
        for i, p in enumerate(cfg.products):
            color = PRODUCT_COLORS[i % len(PRODUCT_COLORS)]
            fig.add_trace(go.Scatter(x=t, y=[s[1][i] for s in plant.samples], name=f'{p.name} stage 1',
                                     line=dict(color=color, dash='dot')))
            fig.add_trace(go.Scatter(x=t, y=[s[2][i] for s in plant.samples], name=f'{p.name} stage 2',
                                     line=dict(color=color)))
    fig.update_layout(title='Buffer levels (dotted: stage-1 buffer, solid: stage-2 buffer)', template='plotly_white',
                      height=380, xaxis_title='hours', yaxis_title='units', margin=dict(l=60, r=20, t=50, b=40))
    return fig


def kpi_cards(env, kpi, hours):
    plant = env.plant
    ops_time = {}
    for r in plant.timeline:
        if r[0].startswith('OP-'):
            ops_time[r[3]] = ops_time.get(r[3], 0) + r[2] - r[1]
    total_op = sum(ops_time.values()) or 1
    items = [
        ('Finished', f"{kpi['finished'] / hours:.0f} units/h", f"capacity {env.cfg.nominal_finish_rate():.0f}"),
        ('Stations starved', f"{100 * kpi['starved_share']:.1f}%", 'of station time'),
        ('Operators working', f"{100 * ops_time.get('work', 0) / total_op:.0f}%",
         f"travel {100 * ops_time.get('travel', 0) / total_op:.0f}% · idle {100 * ops_time.get('idle', 0) / total_op:.0f}%"),
        ('Changeovers', f"{kpi['changeovers']:.0f}", f'over {hours:g} h'),
        ('Reward', f"{kpi['reward']:.1f}", 'episode total'),
    ]
    return [html.Div([html.Div(t, style={'fontSize': '12px', 'color': '#666'}),
                      html.Div(v, style={'fontSize': '22px', 'fontWeight': 600}),
                      html.Div(s, style={'fontSize': '12px', 'color': '#888'})],
                     style={**CARD, 'flex': '1', 'minWidth': '150px', 'marginRight': '10px'}) for t, v, s in items]


# ---------------------------------------------------------------------- layout
def number_field(fid, label):
    return html.Div([html.Label(label, htmlFor=fid, style=LABEL),
                     dcc.Input(id=fid, type='number', min=0, debounce=True, style=INPUT)],
                    style={'flex': '1 1 45%', 'minWidth': '130px', 'margin': '0 8px 8px 0'})


def setup_panel():
    sections = [html.Div([html.H4(title, style={'margin': '0 0 8px'}),
                          html.Div([number_field(fid, label) for fid, _, label in group],
                                   style={'display': 'flex', 'flexWrap': 'wrap'})], style=CARD)
                for title, group in FIELDS]
    return html.Div([
        html.Div([
            html.H4('Plant file', style={'margin': '0 0 8px'}),
            dcc.Dropdown(id='cfg-file', options=config_files(), value='default.json', clearable=False),
            html.Div([dcc.Input(id='cfg-name', placeholder='name to save as', style={**INPUT, 'flex': '1'}),
                      html.Button('Save', id='cfg-save', style={'marginLeft': '6px'})],
                     style={'display': 'flex', 'marginTop': '8px'}),
            html.Div(id='cfg-msg', style={'fontSize': '13px', 'marginTop': '6px'}),
        ], style=CARD),
        *sections,
        html.Div([
            html.H4('Products', style={'margin': '0 0 8px'}),
            dash_table.DataTable(id='products', columns=PRODUCT_COLUMNS, editable=True, row_deletable=True,
                                 style_table={'overflowX': 'auto'},
                                 style_cell={'fontSize': '13px', 'padding': '4px', 'minWidth': '60px'},
                                 style_header={'fontWeight': 600, 'whiteSpace': 'normal'}),
            html.Button('Add product', id='add-product', style={'marginTop': '8px'}),
        ], style=CARD),
    ], style={'width': '440px', 'flexShrink': 0, 'marginRight': '16px'})


def results_panel():
    return html.Div([
        html.Div([
            html.Div(id='capacity', style={'fontSize': '14px', 'marginBottom': '10px'}),
            html.Div([
                html.Div([html.Label('Simulated hours', style=LABEL),
                          dcc.Input(id='sim-hours', type='number', min=1, value=24, style=INPUT)],
                         style={'width': '120px', 'marginRight': '10px'}),
                html.Div([html.Label('Seed', style=LABEL),
                          dcc.Input(id='sim-seed', type='number', min=0, value=0, style=INPUT)],
                         style={'width': '90px', 'marginRight': '10px'}),
                html.Div([html.Label('Policy', style=LABEL),
                          dcc.Dropdown(id='sim-policy', options=policy_options(), value='cover', clearable=False)],
                         style={'width': '240px', 'marginRight': '10px'}),
                html.Button('Run simulation', id='run', style={'height': '34px', 'fontWeight': 600}),
                html.Button('Download event log', id='dl-btn', style={'height': '34px', 'marginLeft': '8px'}),
                dcc.Download(id='dl'),
            ], style={'display': 'flex', 'alignItems': 'flex-end', 'flexWrap': 'wrap'}),
            html.Div(id='run-msg', style={'fontSize': '13px', 'marginTop': '6px', 'color': '#b00'}),
        ], style=CARD),
        dcc.Loading(html.Div([
            html.Div(id='kpis', style={'display': 'flex', 'flexWrap': 'wrap'}),
            dcc.Tabs(id='tabs', value='stations', children=[
                dcc.Tab(label='Finishing stations', value='stations', children=dcc.Graph(id='g-stations', figure=empty_figure())),
                dcc.Tab(label='Machines', value='machines', children=dcc.Graph(id='g-machines', figure=empty_figure())),
                dcc.Tab(label='Operators', value='operators', children=dcc.Graph(id='g-operators', figure=empty_figure())),
                dcc.Tab(label='Buffers', value='buffers', children=dcc.Graph(id='g-buffers', figure=empty_figure())),
            ]),
        ])),
    ], style={'flex': '1', 'minWidth': '0'})


app = Dash(__name__, title='Plant setup')
app.layout = html.Div([
    dcc.Store(id='base-config'),
    html.H2('Plant setup & simulation', style={'margin': '0 0 12px'}),
    html.Div([setup_panel(), results_panel()], style={'display': 'flex', 'alignItems': 'flex-start'}),
], style={'fontFamily': 'system-ui, sans-serif', 'background': '#f5f6f8', 'padding': '16px', 'minHeight': '100vh'})

FIELD_IDS = [fid for fid, _, _ in FIELD_LIST]


# ---------------------------------------------------------------------- callbacks
@app.callback([Output('base-config', 'data'), Output('products', 'data')] + [Output(f, 'value') for f in FIELD_IDS],
              Input('cfg-file', 'value'))
def load_config(name):
    path = os.path.join(CONFIG_DIR, name) if name else None
    cfg = PlantConfig.load(path) if path and os.path.exists(path) else default_config()
    values, rows = config_to_form(cfg)
    return [cfg.to_dict(), rows] + values


@app.callback(Output('products', 'data', allow_duplicate=True), Input('add-product', 'n_clicks'),
              State('products', 'data'), prevent_initial_call=True)
def add_product(_, rows):
    rows = list(rows or [])
    rows.append({'name': f'P{len(rows) + 1}', 'stage1_time': 4.0, 'stage2_time': 2.0, 'finish_time': 20.0, 'share': 0})
    return rows


@app.callback(Output('capacity', 'children'),
              [Input('products', 'data')] + [Input(f, 'value') for f in FIELD_IDS], State('base-config', 'data'))
def show_capacity(rows, *args):
    *values, base = args
    if base is None:
        return ''
    try:
        cfg, warnings = form_to_config(base, values, rows or [])
    except (ValueError, TypeError) as e:
        return html.Span(str(e), style={'color': '#b00'})
    notes = [part for w in warnings for part in (html.Br(), html.Span(w, style={'color': '#b60'}))]
    return html.Span([capacity_summary(cfg)] + notes)


@app.callback([Output('cfg-msg', 'children'), Output('cfg-file', 'options'), Output('cfg-file', 'value')],
              Input('cfg-save', 'n_clicks'),
              [State('cfg-name', 'value'), State('base-config', 'data'), State('products', 'data')] +
              [State(f, 'value') for f in FIELD_IDS], prevent_initial_call=True)
def save_config(_, name, base, rows, *values):
    if not name or not name.replace('-', '').replace('_', '').isalnum():
        return 'Enter a name (letters, digits, - or _)', no_update, no_update
    try:
        cfg, _ = form_to_config(base, values, rows or [])
    except (ValueError, TypeError) as e:
        return str(e), no_update, no_update
    fname = f'{name}.json'
    cfg.save(os.path.join(CONFIG_DIR, fname))
    return f'Saved {fname}', config_files(), fname


@app.callback([Output('kpis', 'children'), Output('g-stations', 'figure'), Output('g-machines', 'figure'),
               Output('g-operators', 'figure'), Output('g-buffers', 'figure'), Output('run-msg', 'children')],
              Input('run', 'n_clicks'),
              [State('base-config', 'data'), State('products', 'data'), State('sim-hours', 'value'),
               State('sim-seed', 'value'), State('sim-policy', 'value')] + [State(f, 'value') for f in FIELD_IDS],
              prevent_initial_call=True)
def run_simulation(_, base, rows, hours, seed, policy_name, *values):
    try:
        cfg, _ = form_to_config(base, values, rows or [])
        hours = float(hours or 24)
        env, kpi = simulate(cfg, hours, int(seed or 0), policy_name)
    except (ValueError, TypeError, RuntimeError) as e:
        return [no_update] * 5 + [str(e)]
    plant = env.plant
    os.makedirs(os.path.dirname(EVENTS_PATH), exist_ok=True)
    plant.log.to_csv(EVENTS_PATH)
    names = [p.name for p in cfg.products]
    stations = [s.name for s in plant.stations]
    machines = [m.name for m in plant.machines]
    operators = [op.name for op in plant.operators]
    station_labels = {s.name: f'{s.name} ({names[s.product]})' for s in plant.stations}
    station_rows = [(station_labels[r[0]],) + r[1:] for r in plant.timeline if r[0] in station_labels]
    return (kpi_cards(env, kpi, hours),
            gantt(station_rows, [station_labels[s] for s in stations], machine_category,
                  'Finishing stations (label shows the product each station is set up for)', names),
            gantt(plant.timeline, machines, machine_category, 'Stage 1 (S1-) and stage 2 (S2-) machines', names),
            gantt(plant.timeline, operators, operator_category, 'Building operators (hover: machine)', names),
            buffer_figure(plant, cfg), '')


@app.callback(Output('dl', 'data'), Input('dl-btn', 'n_clicks'), prevent_initial_call=True)
def download_events(_):
    if not os.path.exists(EVENTS_PATH):
        return no_update
    return dcc.send_file(EVENTS_PATH)


if __name__ == '__main__':
    app.run(debug=False)
