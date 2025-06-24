# flux-conditioning-store

**Repository Name:** FLUX Conditioning Store

**Author:** dwringer

**License:** MIT

**Requirements:**
- invokeai>=4

## Introduction
This is a pair of nodes for taking FLUX Conditioning objects - specifically, prompt embeddings (consisting of T5 and CLIP embedding tensors) - and storing/retrieving them with an SQLite database file. This avoids the overhead of having to encode the conditioning somehow in image metadata, with the disadvantage of being non-portable.

The database is stored in a file, `flux\_conditionings.db`, right alongside the source code of this node. In the `store\_flux\_conditioning.py` file, there are configurable variables to set the maximum DB size and warning thresholds, after which the oldest entries will be automatically deleted. Therefore, if you expect to need them for a while, make sure to increase the size limit.

### Installation:

To install these nodes, simply place the folder containing this
repository's code (or just clone the repository yourself) into your
`invokeai/nodes` folder.

## Overview
### Nodes
- [Retrieve Flux Conditioning](#retrieve-flux-conditioning) - Retrieves one or more FLUX Conditioning objects (CLIP and T5 embeddings)
- [Store Flux Conditioning](#store-flux-conditioning) - Stores a FLUX Conditioning object (CLIP and T5 embeddings) into an SQLite database.

<details>
<summary>

### Functions

</summary>

- `_get_db_size` - Returns the current size of the database file in bytes.
- `_manage_db_size` - Manages the database size, deleting oldest entries if the maximum size is exceeded.
- `_init_db` - Initializes the SQLite database and creates the table for storing conditioning data.
</details>

<details>
<summary>

### Output Definitions

</summary>

- `RetrieveFluxConditioningMultiOutput` - Output definition with 2 fields
- `FluxConditioningStoreOutput` - Output definition with 1 fields
</details>

## Nodes
### Retrieve Flux Conditioning
**ID:** `retrieve_flux_conditioning`

**Category:** conditioning

**Tags:** conditioning, flux, database, retrieve, list, timestamp, deterministic

**Version:** 1.1.4

**Description:** Retrieves one or more FLUX Conditioning objects (CLIP and T5 embeddings)

from an SQLite database using unique identifiers.
    Outputs a single selected conditioning and a list of all retrieved conditionings.
    Includes an option to update the timestamp of retrieved entries.
    The input conditioning IDs are sorted for deterministic retrieval.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `conditioning_id_or_list` | `Union[(str, list[str])]` | The unique identifier(s) of the Flux Conditioning(s) to retrieve. | None |
| `select_index` | `int` | Index of the retrieved conditioning to output as the single 'conditioning' field. If out of bounds, uses modulus. | 0 |
| `touch_timestamp` | `bool` | When true, updates the timestamp of retrieved entries to 'now', preventing early purge. | False |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `RetrieveFluxConditioningMultiOutput`

| Name | Type | Description |
| ---- | ---- | ----------- |
| `conditioning` | `FluxConditioningField` | A single selected Flux Conditioning (selected by index from the retrieved list) |
| `conditioning_list` | `list[FluxConditioningField]` | A list of all retrieved Flux Conditionings |


</details>

---
### Store Flux Conditioning
**ID:** `store_flux_conditioning`

**Category:** conditioning

**Tags:** conditioning, flux, database, store

**Version:** 1.0.0

**Description:** Stores a FLUX Conditioning object (CLIP and T5 embeddings) into an SQLite database.

Returns a unique identifier for retrieval.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `conditioning` | `FluxConditioningField` | The FLUX Conditioning object to store. | None |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `FluxConditioningStoreOutput`

| Name | Type | Description |
| ---- | ---- | ----------- |
| `conditioning_id` | `str` | Unique identifier for the stored Flux Conditioning |


</details>

---

## Footnotes
For questions/comments/concerns/etc, use github or drop into the InvokeAI discord where you'll probably find someone who can help.
