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
- [Concatenate Flux Conditionings](#concatenate-flux-conditionings) - Concatenates the T5 embedding tensors of up to six input Flux Conditioning objects.
- [Flux Conditioning List](#flux-conditioning-list) - Takes multiple optional Flux Conditioning inputs and outputs them as a single
- [Retrieve Flux Conditioning](#retrieve-flux-conditioning) - Retrieves one or more FLUX Conditioning objects (CLIP and T5 embeddings)
- [Store Flux Conditioning](#store-flux-conditioning) - Stores a FLUX Conditioning object (CLIP and T5 embeddings) into an SQLite database.

<details>
<summary>

### Functions

</summary>

- `_get_db_size` - Returns the current size of the database file in bytes.
- `_manage_db_size` - Manages the database size, deleting a fraction of oldest entries if the maximum size is exceeded.
- `_init_db` - Initializes the SQLite database and creates the table for storing conditioning data.
</details>

<details>
<summary>

### Output Definitions

</summary>

- `FluxConditioningListOutput` - Output definition with 1 fields
- `RetrieveFluxConditioningMultiOutput` - Output definition with 2 fields
- `FluxConditioningStoreOutput` - Output definition with 1 fields
</details>

## Nodes
### Concatenate Flux Conditionings
**ID:** `flux_conditioning_concatenate`

**Category:** conditioning

**Tags:** conditioning, flux, concatenate, merge, utility

**Version:** 1.3.0

**Description:** Concatenates the T5 embedding tensors of up to six input Flux Conditioning objects.

Provides flexible control over the CLIP embedding: select by index, concatenate all,
    or generate a pad tensor.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `conditioning_1` | `Any` | First optional Flux Conditioning input. | None |
| `strength_1` | `float` | Strength for the first conditioning input (multiplies its embedding tensors). | 1.0 |
| `conditioning_2` | `Any` | Second optional Flux Conditioning input. | None |
| `strength_2` | `float` | Strength for the second conditioning input (multiplies its embedding tensors). | 1.0 |
| `conditioning_3` | `Any` | Third optional Flux Conditioning input. | None |
| `strength_3` | `float` | Strength for the third conditioning input (multiplies its embedding tensors). | 1.0 |
| `conditioning_4` | `Any` | Fourth optional Flux Conditioning input. | None |
| `strength_4` | `float` | Strength for the fourth conditioning input (multiplies its embedding tensors). | 1.0 |
| `conditioning_5` | `Any` | Fifth optional Flux Conditioning input. | None |
| `strength_5` | `float` | Strength for the fifth conditioning input (multiplies its embedding tensors). | 1.0 |
| `conditioning_6` | `Any` | Sixth optional Flux Conditioning input. | None |
| `strength_6` | `float` | Strength for the sixth conditioning input (multiplies its embedding tensors). | 1.0 |
| `select_clip` | `int` | Index of single CLIP embedding to pass on [0-n).  | 0 |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `FluxConditioningOutput.build(...)`



</details>

---
### Flux Conditioning List
**ID:** `flux_conditioning_list`

**Category:** conditioning

**Tags:** conditioning, flux, list, utility, order

**Version:** 1.0.1

**Description:** Takes multiple optional Flux Conditioning inputs and outputs them as a single

ordered list. Missing (None) inputs are gracefully handled.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `conditioning_1` | `Any` | First optional Flux Conditioning input. | None |
| `conditioning_2` | `Any` | Second optional Flux Conditioning input. | None |
| `conditioning_3` | `Any` | Third optional Flux Conditioning input. | None |
| `conditioning_4` | `Any` | Fourth optional Flux Conditioning input. | None |
| `conditioning_5` | `Any` | Fifth optional Flux Conditioning input. | None |
| `conditioning_6` | `Any` | Sixth optional Flux Conditioning input. | None |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `FluxConditioningListOutput`

| Name | Type | Description |
| ---- | ---- | ----------- |
| `conditioning_list` | `list[FluxConditioningField]` | An ordered list of provided Flux Conditioning objects. |


</details>

---
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

**Version:** 1.0.1

**Description:** Stores a FLUX Conditioning object (CLIP and T5 embeddings) into an SQLite database.

Returns a unique identifier for retrieval.
    Includes database size management with proactive deletion and VACUUM.

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
