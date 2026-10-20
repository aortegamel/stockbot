# add_widget

Domain: market
Family: workspace
Intent: add_widget
Output kind: widget_selection
Source: local
Entity scope: single_security
Time mode: current

Add one widget to the stock workspace.

## Choose when

- The user asks to add a chart, table, or data widget to a stock workspace.

## Reject when

- The receipt is not factual research evidence. Not for questions about the widget data.

## Conflicts with

None

## Related tools

None

## Prerequisites

None

## Required arguments

- `ticker` (string)
- `type` (string)

## Optional arguments

None
