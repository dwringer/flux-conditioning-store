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
- [Extract Image Collection Metadata Item](#extract-image-collection-metadata-item) - This node extracts specified metadata values from a collection of input images.
- [Flux Conditioning Blend](#flux-conditioning-blend) - Performs a blend between two FLUX Conditioning objects using either direct SLERP
- [Flux Conditioning Delta](#flux-conditioning-delta) - Calculates the delta between feature and reference conditionings,
- [Flux Conditioning List](#flux-conditioning-list) - Takes multiple optional Flux Conditioning inputs and outputs them as a single
- [Retrieve Flux Conditioning](#retrieve-flux-conditioning) - Retrieves one or more FLUX Conditioning objects (CLIP and T5 embeddings)
- [Store Flux Conditioning](#store-flux-conditioning) - Stores a FLUX Conditioning object (CLIP and T5 embeddings) into an SQLite database.

<details>
<summary>

### Functions

</summary>

- `_average_conditioning_list` - Averages a list of FLUXConditioningInfo objects into a single one.
- `slerp` - Performs spherical linear interpolation (SLERP) between two *normalized* tensors.
- `_get_db_size` - Returns the current size of the database file in bytes.
- `_manage_db_size` - Manages the database size, deleting a fraction of oldest entries if the maximum size is exceeded.
- `_init_db` - Initializes the SQLite database and creates the table for storing conditioning data.
</details>

<details>
<summary>

### Output Definitions

</summary>

- `FluxConditioningDeltaAndAugmentedOutput` - Output definition with 2 fields
- `FluxConditioningBlendOutput` - Output definition with 1 fields
- `FluxConditioningListOutput` - Output definition with 1 fields
- `RetrieveFluxConditioningMultiOutput` - Output definition with 2 fields
- `FluxConditioningStoreOutput` - Output definition with 1 fields
</details>

## Nodes
### Concatenate Flux Conditionings
**ID:** `flux_conditioning_concatenate`

**Category:** conditioning

**Tags:** conditioning, flux, concatenate, merge, utility

**Version:** 1.4.1

**Description:** Concatenates the T5 embedding tensors of up to six input Flux Conditioning objects.

Provides flexible control over the CLIP embedding: select by 1-indexed input number,
    or generate a zeros tensor.

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
| `select_clip` | `int` | CLIP embedding selection: 0 for a zeros tensor; 1-6 to select a specific input (1-indexed). If a selected input is missing, it falls back to the next subsequent, then preceding, available CLIP embedding. | 1 |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `FluxConditioningOutput.build(...)`



</details>

---
### Extract Image Collection Metadata Item
**ID:** `extract_image_collection_metadata_item`

**Category:** metadata

**Tags:** image, metadata, extraction, collection, utility

**Version:** 1.0.0

**Description:** This node extracts specified metadata values from a collection of input images.

It takes an image collection and a metadata key string input.
    For each image in the collection, it attempts to retrieve the value associated
    with the provided key. The extracted values are then compiled into a string 
    collection. If a key is not found for a particular image, an empty string is 
    appended to maintain collection length.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `images` | `list[ImageField]` | A collection of images from which to extract metadata. | None |
| `key` | `str` | Metadata key to extract values for Output. Leave empty to ignore. |  |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `StringCollectionOutput`



</details>

---
### Flux Conditioning Blend
**ID:** `flux_conditioning_blend`

**Category:** conditioning

**Tags:** conditioning, flux, interpolation, slerp, lerp, blend

**Version:** 1.0.0

**Description:** Performs a blend between two FLUX Conditioning objects using either direct SLERP

or an advanced method that separates magnitude and direction.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `conditioning_1` | `FluxConditioningField` | The first FLUX Conditioning object. | None |
| `conditioning_2` | `FluxConditioningField` | The second FLUX Conditioning object. | None |
| `alpha` | `float` | Interpolation factor (0.0 for conditioning_1, 1.0 for conditioning_2). | 0.5 |
| `use_magnitude_separation` | `bool` | If True, uses magnitude separation (SLERP for direction, LERP for magnitude); otherwise, uses direct SLERP. | False |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `FluxConditioningBlendOutput`

| Name | Type | Description |
| ---- | ---- | ----------- |
| `conditioning` | `FluxConditioningField` | The interpolated Flux Conditioning |


</details>

---
### Flux Conditioning Delta
**ID:** `flux_conditioning_delta_augmentation`

**Category:** conditioning

**Tags:** conditioning, flux, arithmetic, delta, augment

**Version:** 1.0.0

**Description:** Calculates the delta between feature and reference conditionings,

and optionally augments a base conditioning with this delta.
    If reference conditioning is omitted, it will be treated as zero tensors.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `feature_conditioning` | `Any` | Feature Conditioning (single or list) for delta calculation. If a list, it will be averaged. | None |
| `reference_conditioning` | `Any` | Reference Conditioning (single or list) for delta calculation. If a list, it will be averaged. If omitted, zero tensors will be used as reference. | None |
| `base_conditioning` | `Any` | Optional Base Conditioning to which the delta will be added. If not provided, Augmented Conditioning will be the Delta. | None |
| `base_scale` | `float` | Scalar to multiply the base conditioning when augmenting. | 1.0 |
| `delta_scale` | `float` | Scalar to multiply the delta when augmenting the base conditioning. | 1.0 |
| `scale_delta_output` | `bool` | If true, the delta output will also be scaled by the delta_scale. | False |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `FluxConditioningDeltaAndAugmentedOutput`

| Name | Type | Description |
| ---- | ---- | ----------- |
| `augmented_conditioning` | `FluxConditioningField` | The augmented conditioning (base + delta, or just delta) |
| `delta_conditioning` | `FluxConditioningField` | The resulting conditioning delta (feature - reference) |


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
