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