# Usage



## Reading and writing a zarr hieararchy

### Reading
The `GroupSpec` and `ArraySpec` classes represent Zarr groups and arrays, respectively. To create an instance of a `GroupSpec` or `ArraySpec` from an existing Zarr group or array, pass the Zarr group / array to the `.from_zarr` method defined on the `GroupSpec` / `ArraySpec` classes. This will result in a `pydantic-zarr` model of the Zarr object. 

Note that `GroupSpec.from_zarr(zarr_group)` will traverse the entire hierarchy under `zarr_group`. Future versions of this library may introduce a limit on the depth of this traversal: see [#2](https://github.com/d-v-b/pydantic-zarr/issues/2).

Note that `from_zarr` will *not* read the data inside an array.

### Writing
To write a hierarchy to some zarr-compatible storage backend, `GroupSpec` and `ArraySpec` have `to_zarr` methods that take a Zarr store and a path and return a Zarr array or group created in the store at the given path. 

Note that `to_zarr` will *not* write any array data. You have to do this separately.

```python
from zarr import group
from zarr.creation import create
from zarr.storage import MemoryStore
from pydantic_zarr import GroupSpec

# create an in-memory Zarr group + array with attributes
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

# modify the spec to define a new Zarr hierarchy
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

### Creating from an array

The `ArraySpec` class has a `from_array` static method that takes a numpy-array-like object and returns an `ArraySpec` with `shape` and `dtype` fields matching those of the array-like object.

```python
from pydantic_zarr import ArraySpec
import numpy as np

print(ArraySpec.from_array(np.arange(10)).dict())
"""
{
    'zarr_version': 2,
    'attrs': {},
    'shape': (10,),
    'chunks': (10,),
    'dtype': '<i8',
    'fill_value': 0,
    'order': 'C',
    'filters': None,
    'dimension_separator': '/',
    'compressor': None,
}
"""
```

## Using generic types  

The following examples demonstrate how to specialize `GroupSpec` and `ArraySpec` with type parameters. By specializing `GroupSpec` or `ArraySpec` in this way, python type checkers and Pydantic can type-check elements of a Zarr hierarchy.

```python
from pydantic_zarr import GroupSpec, ArraySpec
from pydantic_zarr.core import TItem, TAttr
from pydantic import ValidationError
from typing import Any, TypedDict

# a Pydantic BaseModel would also work here
class GroupAttrs(TypedDict):
    a: int
    b: int

# a Zarr group with attributes consistent with GroupAttrs
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

# a Zarr group that only contains arrays -- no subgroups!
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
print(ArraysOnlyGroup(attrs={}, items=items).dict())
"""
{
    'zarr_version': 2,
    'attrs': {},
    'items': {
        'foo': {
            'zarr_version': 2,
            'attrs': {},
            'shape': (1,),
            'chunks': (1,),
            'dtype': '|u1',
            'fill_value': 0,
            'order': 'C',
            'filters': None,
            'dimension_separator': '/',
            'compressor': None,
        }
    },
}
"""
```