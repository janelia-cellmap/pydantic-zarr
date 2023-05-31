# pydantic-zarr
[Pydantic](https://docs.pydantic.dev/1.10/) models for [Zarr](https://zarr.readthedocs.io/en/stable/index.html). Not stable. Do not use in production.

## About
This library uses pydantic models to define a storage-independent JSON-serializable model of a [Zarr](https://zarr.readthedocs.io/en/stable/index.html) hierarchy, i.e. a tree of groups and arrays. This representation of the hierarchy can be derived from, and serialized to, any zarr store. These models can also be validated.


