"""Structured log of everything that happens in the plant.

One row per event: (t, event, entity, product, qty, detail)
  t        minutes since episode start
  event    see EVENTS
  entity   machine / station / operator id, e.g. 'S1-03', 'S2-11', 'F-07', 'OP-04', 'prep'
  product  product name or ''
  qty      units involved (0 when not relevant)
  detail   free text (reason, destination, ...)
"""
import csv

EVENTS = (
    'decision',          # agent set a machine's product/priority (detail = 'prio=<p>')
    'op_dispatch',       # operator leaves for a machine (detail = destination)
    'op_arrive',         # operator starts working on a machine
    'op_release',        # operator leaves a machine (detail = reason)
    'op_idle',           # operator found no task
    'changeover_start', 'changeover_end',
    'unit_start', 'unit_done',
    'consume',           # components or intermediate units taken (detail = source)
    'store',             # unit put in a buffer (detail = buffer)
    'starved',           # machine/station waiting on inputs (detail = what is missing)
    'blocked',           # machine cannot start: output buffer full
    'kit_request', 'kit_delivered', 'kit_discarded',
    'breakdown', 'repaired',
    'load',              # finishing station loaded (qty units)
    'finish_done',       # finishing cycle complete (qty units)
)

COLUMNS = ('t', 'event', 'entity', 'product', 'qty', 'detail')


class EventLog:
    def __init__(self, enabled=True):
        self.enabled = enabled   # turned off during training: a 72 h episode logs ~10^5 rows
        self.rows = []

    def __call__(self, t, event, entity, product='', qty=0, detail=''):
        if self.enabled:
            self.rows.append((round(t, 3), event, entity, product, qty, detail))

    def clear(self):
        self.rows = []

    def to_csv(self, path):
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            w.writerows(self.rows)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.rows, columns=COLUMNS)
