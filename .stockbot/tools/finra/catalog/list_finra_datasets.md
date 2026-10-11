# list_finra_datasets

Domain: finra
Family: catalog
Intent: discover_finra_dataset
Output kind: catalog
Source: finra
Entity scope: dataset_catalog
Time mode: current

Catalog of public FINRA datasets with canonical ids, groups, and ticker/date support.

## Choose when

- Finding which FINRA dataset covers a question before querying.

## Reject when

- Does not return dataset fields or schemas (describe_finra_dataset).
- Unneeded when the dataset ID is known. Call the analysis tool directly.

## Conflicts with

- describe_finra_dataset

## Related tools

- describe_finra_dataset

## Prerequisites

None

## Required arguments

None

## Optional arguments

- `group` (string): Optional dataset group filter (e.g. otcMarket, fixedIncomeMarket, finra).
- `search` (string): Optional substring match on name/description.
