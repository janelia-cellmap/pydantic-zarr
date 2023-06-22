# pydantic-zarr
[Pydantic](https://docs.pydantic.dev/1.10/) models for [Zarr](https://zarr.readthedocs.io/en/stable/index.html). Not stable. Do not use in production.

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
grp = group(store=store, path=)
grp.attrs.put({'foo': 10})
arr = create(path='foo/bar',store=store, shape=(10,), dtype='float32')
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
      "dtype": "float32",
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
print(ArraysOnlyGroup(attrs={}, items={'foo': ArraySpec(attrs={}, shape=(1,), dtype='uint8', chunks=(1,))}))
#> zarr_version=2 attrs={} items={'foo': ArraySpec(zarr_version=2, attrs={}, shape=(1,), chunks=(1,), dtype='uint8', fill_value=0, order='C', filters=None, dimension_separator='/', compressor=None)}
```