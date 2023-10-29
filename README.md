# pydantic-zarr

[![PyPI](https://img.shields.io/pypi/v/pydantic-zarr)](https://pypi.python.org/pypi/pydantic-zarr)

[Pydantic](https://docs.pydantic.dev/1.10/) models for [Zarr](https://zarr.readthedocs.io/en/stable/index.html).

## ⚠️ Disclaimer ⚠️
This project is under a lot of flux -- I want to add [zarr version 3](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html) support to this project, but the [reference python implementation](https://github.com/zarr-developers/zarr-python) doesn't support version 3 yet. Also, the key ideas in this repo may change in the process of being formalized over in [this specification](https://github.com/zarr-developers/zeps/pull/46) (currently just a draft). As the ecosystem evolves I *will* be breaking things (and versioning the project accordingly), so be advised!

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
print(spec.model_dump())
"""
{
    'zarr_version': 2,
    'attributes': {},
    'members': {
        'bar': {
            'zarr_version': 2,
            'attributes': {'metadata': 'hello'},
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
