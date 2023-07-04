# pydantic-zarr
[Pydantic](https://docs.pydantic.dev/1.10/) models for [Zarr](https://zarr.readthedocs.io/en/stable/index.html). Under active development, expect breaking changes!

## About
This library uses pydantic models to define a storage-independent JSON-serializable model of a [Zarr](https://zarr.readthedocs.io/en/stable/index.html) hierarchy, i.e. a tree of groups and arrays. This representation of the hierarchy can be derived from, and serialized to, any zarr store. These models can also be validated.

## Examples

Represent an existing zarr hierarchy in a storage-independent manner:

```python
from zarr import group
from zarr.creation import create
from zarr.storage import MemoryStore
from pydantic_zarr.core import GroupSpec

# create an in-memory zarr group + array with attributes
store = MemoryStore()

# create a group at the path 'foo'
grp = group(store=store, path='foo')
grp.attrs.put({'foo': 10})

# create an array called 'bar' inside 'foo'
arr = create(path='foo/bar', store=store, shape=(10,))
arr.attrs.put({'array_metadata': True})

spec = GroupSpec.from_zarr(grp)
print(spec.json(indent=2))
"""
{
  "zarr_version": 2,
  "attrs": {
    "foo": 10
  },
  "items": {
    "foo": {
      "zarr_version": 2,
      "attrs": {},
      "items": {
        "bar": {
          "zarr_version": 2,
          "attrs": {
            "array_metadata": true
          },
          "shape": [
            10
          ],
          "chunks": [
            10
          ],
          "dtype": "<f8",
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
  }
}
"""
# modify the spec to define a new zarr hierarchy
spec2 = spec.copy()
spec2.attrs = {'a': 100, 'b': 'metadata'}
spec2.items['bar'].shape = (100,)

# serialize the spec to the store
group2 = spec2.to_zarr(store, path='foo2')

print(group2)
#> <zarr.hierarchy.Group '/foo2'>

print(dict(group2.attrs))
#> {'a': 100, 'b': 'metadata'}

print(group2['bar'])
#> <zarr.core.Array '/foo2/bar' (100,) float32>

print(dict(group2['bar'].attrs))
#> {'array_metadata': True}
```

Use type annotations to restrict the structure of group attributes and group contents

```python
from pydantic_zarr import GroupSpec, ArraySpec
from pydantic import ValidationError
from typing import Any, TypedDict

class GroupAttrs(TypedDict):
    a: int
    b: int

# specify a zarr group that must have specific attributes
SpecificAttrsGroup = GroupSpec[GroupAttrs, Any]

try:
    SpecificAttrsGroup(attrs={'a' : 10, 'b': 'foo'})
except ValidationError as exc:
    print(exc)
    """
    1 validation error for GroupSpec[GroupAttrs, Any]
    attrs -> b
        value is not a valid integer (type=type_error.integer)
    """
# this passes validation
print(SpecificAttrsGroup(attrs={'a': 100, 'b': 100}))
#> zarr_version=2 attrs={'a': 100, 'b': 100} items={}

# specify a zarr group that can only contain arrays, not other groups
ArraysOnlyGroup = GroupSpec[Any, ArraySpec]

try:
    ArraysOnlyGroup(attrs={}, items={'foo': GroupSpec(attrs={})})
except ValidationError as exc:
    print(exc)
    """
    4 validation errors for GroupSpec[Any, ArraySpec]
    items -> foo -> shape
        field required (type=value_error.missing)
    items -> foo -> chunks
        field required (type=value_error.missing)
    items -> foo -> dtype
        field required (type=value_error.missing)
    items -> foo -> items
        extra fields not permitted (type=value_error.extra)
    """

# this passes validation
items = items={'foo': ArraySpec(attrs={}, shape=(1,), dtype='uint8', chunks=(1,))}
print(ArraysOnlyGroup(attrs={}, items = items))
"""
{
  "zarr_version": 2,
  "attrs": {},
  "items": {
    "foo": {
      "zarr_version": 2,
      "attrs": {},
      "shape": [
        1
      ],
      "chunks": [
        1
      ],
      "dtype": "|u1",
      "fill_value": 0,
      "order": "C",
      "filters": null,
      "dimension_separator": "/",
      "compressor": null
    }
  }
}
"""
```

Python's type system can only take you so far. Pydantic validators can be used to apply
even further restrictions on zarr hierarchies.

```python
from pydantic import validator
from pydantic_zarr import GroupSpec, ArraySpec
import numpy as np

# define an array that must have uint8 dtype
class Uint8Array(ArraySpec):

    @validator('dtype')
    def dtype_is_uint8(cls, v):
        """
        Raises a ValueError if the dtype is not '|u1' (uint8)
        """
        if v != "|u1":
            msg = f"dtype must be '|u1' (uint8); got {v} instead."
            raise ValueError(msg)
        return v
    
# this will fail validation
try:
    Uint8Array(attrs={}, shape=(10,), chunks=(10,), dtype='float32')
except ValidationError as exc:
    print(exc)
"""
1 validation error for Uint8Array
dtype
  dtype must be '|u1' (uint8); got <f4 instead. (type=value_error)
"""

# this passes
print(Uint8Array(attrs={}, shape=(10,), chunks=(10,), dtype='uint8').json(indent=2))
"""
{
  "zarr_version": 2,
  "attrs": {},
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
  "dimension_separator": "/",
  "compressor": null
}
"""

# define a group that can only contain arrays with the same dtype
class SameDtypeGroup(GroupSpec[Any, ArraySpec[Any]]):

    @validator('items')
    def dtypes_are_uniform(cls, v):
        """
        Raises a ValueError if the dtypes of the arrays are not the same
        """
        dtypes = set(arr.dtype for arr in v.values())
        if len(dtypes) > 1:
            msg = (f"Got arrays with multiple dtypes: {dtypes}. "
                   "Arrays must have the same dtype.")
            raise ValueError(msg)
        return v


# this will fail validation
try:
    items = {
        'array_a': ArraySpec.from_array(np.arange(10)), 
        'array_b': ArraySpec.from_array(np.arange(10, dtype='uint8'))}
    SameDtypeGroup(attrs={}, items=items)
except ValidationError as exc:
    print(exc)
"""
1 validation error for SameDtypeGroup
items
  Got arrays with multiple dtypes: {'|u1', '<i8'}. Arrays must have the same dtype. (type=value_error)
"""

# this passes
items = {
        'array_a': ArraySpec.from_array(np.arange(10, dtype='uint8')), 
        'array_b': ArraySpec.from_array(np.arange(10, dtype='uint8'))}

print(SameDtypeGroup(attrs={}, items=items).json(indent=2))
"""
{
  "zarr_version": 2,
  "attrs": {},
  "items": {
    "array_a": {
      "zarr_version": 2,
      "attrs": {},
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
      "dimension_separator": "/",
      "compressor": null
    },
    "array_b": {
      "zarr_version": 2,
      "attrs": {},
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
      "dimension_separator": "/",
      "compressor": null
    }
  }
}
"""

```