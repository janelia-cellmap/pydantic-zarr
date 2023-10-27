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
from pydantic_zarr.v2 import GroupSpec

# create an in-memory Zarr group + array with attributes
grp = group(path='foo')
grp.attrs.put({'group_metadata': 10})
arr = create(path='foo/bar', store=grp.store, shape=(10,), compressor=None)
arr.attrs.put({'array_metadata': True})

spec = GroupSpec.from_zarr(grp)
spec_dict = spec.model_dump()
print(spec_dict)
"""
{
    'zarr_version': 2,
    'attributes': {'group_metadata': 10},
    'members': {
        'bar': {
            'zarr_version': 2,
            'attributes': {'array_metadata': True},
            'shape': (10,),
            'chunks': (10,),
            'dtype': '<f8',
            'fill_value': 0.0,
            'order': 'C',
            'filters': None,
            'dimension_separator': '.',
            'compressor': None,
        }
    },
}
"""

# modify the spec to define a new Zarr hierarchy
spec_dict2 = spec_dict.copy()
spec_dict2['attributes'] = {'a': 100, 'b': 'metadata'}
spec_dict2['members']['bar']['shape'] = [100,]
spec2 = GroupSpec(**spec_dict2)
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

print(ArraySpec.from_array(np.arange(10)).model_dump())
"""
{
    'zarr_version': 2,
    'attributes': {},
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
class MyAttrs(TypedDict):
    a: int
    b: int

# a Zarr group with attributes consistent with MyAttrs
MyAttrsGroup = GroupSpec[MyAttrs, TItem]

try:
    MyAttrsGroup(attributes={'a' : 10, 'b': 'foo'})
except ValidationError as exc:
    print(exc)
    """
    1 validation error for GroupSpec[MyAttrs, TItem]
    attributes -> b
      value is not a valid integer (type=type_error.integer)
    """

# this passes validation
print(MyAttrsGroup(attributes={'a': 100, 'b': 100}))
#> zarr_version=2 attributes={'a': 100, 'b': 100} members={}

# a Zarr group that only contains arrays -- no subgroups!
# we re-use the Tattributes type variable defined in pydantic_zarr.core
ArraysOnlyGroup = GroupSpec[TAttr, ArraySpec]

try:
    ArraysOnlyGroup(attributes={}, members={'foo': GroupSpec(attributes={})})
except ValidationError as exc:
    print(exc)
    """
    4 validation errors for GroupSpec[TAttr, ArraySpec]
    members -> foo -> shape
      field required (type=value_error.missing)
    members -> foo -> chunks
      field required (type=value_error.missing)
    members -> foo -> dtype
      field required (type=value_error.missing)
    members -> foo -> members
      extra fields not permitted (type=value_error.extra)
    """

# this passes validation
members = {'foo': ArraySpec(attributes={}, 
                            shape=(1,), 
                            dtype='uint8', 
                            chunks=(1,), 
                            compressor=None)}
print(ArraysOnlyGroup(attributes={}, members=members).model_dump())
"""
{
    'zarr_version': 2,
    'attributes': {},
    'members': {
        'foo': {
            'zarr_version': 2,
            'attributes': {},
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