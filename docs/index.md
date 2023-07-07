# pydantic-zarr

Bringing static typing and runtime validation to zarr hiearchies.

## Overview
`pydantic-zarr` expresses data stored in the [zarr](https://zarr.readthedocs.io/en/stable/) format with [pydantic](https://docs.pydantic.dev/1.10/). Specifically, `pydantic-zarr` encodes zarr groups and arrays as [pydantic models](https://docs.pydantic.dev/1.10/usage/models/). Programmers can use these models formalize the structure of zarr hieararchies, enabling type-checking and runtime validation of zarr data. 

```python
import zarr
from pydantic_zarr import GroupSpec
group = zarr.group(path='foo')
array = zarr.create(store = group.store, path='foo/bar', shape=10, dtype='uint8')
array.attrs.put({'metadata': 'hello'})

# this is a pydantic model
spec = GroupSpec.from_zarr(group)
print(spec.json(indent=2))
"""
{
  "zarr_version": 2,
  "attrs": {},
  "items": {
    "bar": {
      "zarr_version": 2,
      "attrs": {
        "metadata": "hello"
      },
      "shape": [
        10
      ],
      "chunks": [
        10
      ],
      "dtype": "|u1",
      "fill_value": 0,
      "order": "C",
      "filters": null,
      "dimension_separator": ".",
      "compressor": {
        "id": "blosc",
        "cname": "lz4",
        "clevel": 5,
        "shuffle": 1,
        "blocksize": 0
      }
    }
  }
}
"""
```


Important note: this library only provides tools to represent the layout of zarr groups and arrays, and their attributes. It performs no type checking or runtime validation of the multidimensional array data contained inside zarr arrays.


## Installation

`pip install -U pydantic-zarr` 

## Design

A Zarr group can be represented as two elements: 

- `attrs`: Anything JSON serializable, typically dict-like with string keys.
- `items`: dict-like, with keys that are strings and values that are other Zarr groups, or Zarr arrays.

A Zarr array can be represented similarly (minus the `items` property).

Accordingly, in `pydantic-zarr`, Zarr groups are encoded by the `GroupSpec` class with two fields:

- `GroupSpec.attrs`: either a `Mapping` or a `pydantic.BaseModel`. 
- `GroupSpec.items`: a mapping with string keys and values that must be `GroupSpec` or `ArraySpec` instances.

Zarr arrays are represented by the `ArraySpec` class, which has a similar `attrs` field, as well as fields for all the zarr array properties (`dtype`, `shape`, `chunks`, etc).

`GroupSpec` and `ArraySpec` are both [generic models](https://docs.pydantic.dev/1.10/usage/models/#generic-models). `GroupSpec` takes two type parameters, the first specializing the type of `GroupSpec.attrs`, and the second specializing the type of the *values* of `GroupSpec.items` (they keys of `GroupSpec.items` are always strings). `ArraySpec` only takes one type parameter, which specializes the type of `ArraySpec.attrs`.

Examples using this generic typing functionality can be found in the [usage guide](intro.md#using-generic-types)