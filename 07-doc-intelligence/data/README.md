# Data Directory

Contains regulatory reference data used by the compliance checker.

## regulations/

- `ghs_sentences.json` — GHS hazard statements (H-codes) used for SDS completeness validation.

## Usage

The `RegulationDB` service loads this data at startup. You can extend the datasets by editing the JSON files or by loading custom regulation packs via the `data_path` parameter.
