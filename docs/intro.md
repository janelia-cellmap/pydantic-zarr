# Introduction



## Example: reading and writing a zarr hieararchy

```python
from zarr import group
from zarr.creation import create
from zarr.storage import MemoryStore
from pydantic_zarr import GroupSpec

# create an in-memory zarr group + array with attributes
grp = group(path='foo')
grp.attrs.put({'group_metadata': 10})
arr = create(path='foo/bar', store=grp.store, shape=(10,), compressor=None)
arr.attrs.put({'array_metadata': True})

spec = GroupSpec.from_zarr(grp)
print(spec.dict())
"""
{
    'zarr_version': 2,
    'attrs': {'group_metadata': 10},
    'items': {
        'bar': {
            'zarr_version': 2,
            'attrs': {'array_metadata': True},
            'shape': (10,),
            'chunks': (10,),
            'dtype': '<f8',
            'fill_value': 0,
            'order': 'C',
            'filters': None,
            'dimension_separator': '.',
            'compressor': None,
        }
    },
}
"""

# modify the spec to define a new zarr hierarchy
spec2 = spec.copy()
spec2.attrs = {'a': 100, 'b': 'metadata'}

spec2.items['bar'].shape = (100,)

# serialize the spec to the store
group2 = spec2.to_zarr(grp.store, path='foo2')

print(group2)
#> <zarr.hierarchy.Group '/foo2'>

print(dict(group2.attrs))
#> {'a': 100, 'b': 'metadata'}

print(group2['bar'])
#> <zarr.core.Array '/foo2/bar' (100,) float64>

print(dict(group2['bar'].attrs))
#> {'array_metadata': True}
```

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

The following exmaples demonstrate how to specialize `GroupSpec` and `ArraySpec` with type parameters.

```python
from pydantic_zarr import GroupSpec, ArraySpec
from pydantic_zarr.core import TItem, TAttr
from pydantic import ValidationError
from typing import Any, TypedDict

# a pydantic BaseModel would also work here
class GroupAttrs(TypedDict):
    a: int
    b: int

# a zarr group with attributes consistent with GroupAttrs
SpecificAttrsGroup = GroupSpec[GroupAttrs, TItem]

try:
    SpecificAttrsGroup(attrs={'a' : 10, 'b': 'foo'})
except ValidationError as exc:
    print(exc)
    """
    1 validation error for GroupSpec[GroupAttrs, TItem]
    attrs -> b
      value is not a valid integer (type=type_error.integer)
    """

# this passes validation
print(SpecificAttrsGroup(attrs={'a': 100, 'b': 100}))
#> zarr_version=2 attrs={'a': 100, 'b': 100} items={}

# a zarr group that only contains arrays -- no subgroups!
# we re-use the TAttrs type variable defined in pydantic_zarr.core
ArraysOnlyGroup = GroupSpec[TAttr, ArraySpec]

try:
    ArraysOnlyGroup(attrs={}, items={'foo': GroupSpec(attrs={})})
except ValidationError as exc:
    print(exc)
    """
    4 validation errors for GroupSpec[TAttr, ArraySpec]
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
items = {'foo': ArraySpec(attrs={}, 
                          shape=(1,), 
                          dtype='uint8', 
                          chunks=(1,), 
                          compressor=None)}
print(ArraysOnlyGroup(attrs={}, items=items).json(indent=2))
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

## Creation

Both `ArraySpec` and `GroupSpec` have static `from_zarr` methods, which take a zarr array or group as arguments and return an `ArraySpec` or `GroupSpec`, respectively.

```python
from zarr import MemoryStore, group
from pydantic_zarr import GroupSpec

store = MemoryStore()
grp = group(store=store, path='foo')
spec = GroupSpec.from_zarr(grp)
```

The `ArraySpec` class has a `from_array` static method that takes a numpy-array-like object and returns an `ArraySpec` with `shape` and `dtype` fields matching those of the array-like object.

```python
from pydantic_zarr import ArraySpec
import numpy as np

print(ArraySpec.from_array(np.arange(10)).json(indent=2))
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
  "dtype": "<i8",
  "fill_value": 0,
  "order": "C",
  "filters": null,
  "dimension_separator": "/",
  "compressor": null
}
"""
```