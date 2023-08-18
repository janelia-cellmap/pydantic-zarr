# pydantic-zarr

[![PyPI](https://img.shields.io/pypi/v/pydantic-zarr)](https://pypi.python.org/pypi/pydantic-zarr)

[Pydantic](https://docs.pydantic.dev/1.10/) models for [Zarr](https://zarr.readthedocs.io/en/stable/index.html).

## Installation

`pip install -U pydantic-zarr`

## Help

See the [documentation](https://janelia-cellmap.github.io/pydantic-zarr/) for detailed information about this project. 

## Example

```python
import zarr
from pydantic_zarr import GroupSpec

group = zarr.group(path='foo')
array = zarr.create(store = group.store, path='foo/bar', shape=10, dtype='uint8')
array.attrs.put({'metadata': 'hello'})

# this is a pydantic model
spec = GroupSpec.from_zarr(group)
print(spec.dict())
"""
{
    'zarr_version': 2,
    'attrs': {},
    'items': {
        'bar': {
            'zarr_version': 2,
            'attrs': {'metadata': 'hello'},
            'shape': (10,),
            'chunks': (10,),
            'dtype': '|u1',
            'fill_value': 0,
            'order': 'C',
            'filters': None,
            'dimension_separator': '.',
            'compressor': {
                'id': 'blosc',
                'cname': 'lz4',
                'clevel': 5,
                'shuffle': 1,
                'blocksize': 0,
            },
        }
    },
}
"""
```
