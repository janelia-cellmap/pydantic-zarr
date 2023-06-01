from __future__ import annotations

from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
)

from pydantic.generics import GenericModel
from zarr.storage import init_group, BaseStore
import zarr
import os

TAttrs = TypeVar("TAttrs", bound=dict[str, Any])
TItem = TypeVar("TItem", bound=Union["ArraySpec", "GroupSpec"])

DimensionSeparator = Union[Literal["."], Literal["/"]]
ZarrVersion = Union[Literal[2], Literal[3]]
ArrayOrder = Union[Literal["C"], Literal["F"]]


class NodeSpec(GenericModel, Generic[TAttrs]):
    zarr_version: ZarrVersion = 2
    attrs: TAttrs


class ArraySpec(NodeSpec, Generic[TAttrs]):
    """
    This pydantic model represents the properties of a zarr array. It is generic
    with respect to the type of attrs.
    """

    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: str
    fill_value: Union[None, int, float] = 0
    order: ArrayOrder = "C"
    filters: Optional[list[dict[str, Any]]] = None
    dimension_separator: DimensionSeparator = "/"
    compressor: Optional[dict[str, Any]] = None

    @classmethod
    def from_zarr(cls, zarray: zarr.Array):

        filters = zarray.filters
        if filters is not None:
            filters = [f.get_config() for f in filters]

        return cls(
            shape=zarray.shape,
            chunks=zarray.chunks,
            dtype=str(zarray.dtype),
            fill_value=zarray.fill_value,
            order=zarray.order,
            filters=filters,
            dimension_separator=zarray._dimension_separator,
            compressor=zarray.compressor.get_config(),
            attrs=dict(zarray.attrs),
        )

    def to_zarr(self, store, path) -> zarr.Array:
        spec_dict = self.dict()
        attrs = spec_dict.pop("attrs")
        result = zarr.create(store=store, path=path, **spec_dict)
        result.attrs.put(attrs)
        return result


class GroupSpec(NodeSpec, Generic[TAttrs, TItem]):
    items: dict[str, TItem] = {}

    @classmethod
    def from_zarr(cls, zgroup: zarr.Group) -> "GroupSpec":
        result: GroupSpec
        items = {}
        for name, item in zgroup.items():
            if isinstance(item, zarr.Array):
                _item = ArraySpec.from_zarr(item)
            elif isinstance(item, zarr.Group):
                _item = cls.from_zarr(item)
            items[name] = _item

        result = GroupSpec(attrs=dict(zgroup.attrs), items=items)
        return result

    def to_zarr(self, store: BaseStore, path: str):
        spec_dict = self.dict()
        # pop items because it's not a valid kwarg for init_group
        spec_dict.pop("items")
        # pop attrs because it's not a valid kwarg for init_group
        attrs = spec_dict.pop("attrs")
        # needing to call init_group, then zarr.group is not ergonomic
        init_group(store=store, path=path)
        result = zarr.group(store=store, path=path, **spec_dict)
        result.attrs.put(attrs)
        for name, item in self.items.items():
            subpath = os.path.join(path, name)
            item.to_zarr(store, subpath)
        return result


def to_spec(element: Union[zarr.Array, zarr.Group]) -> Union[ArraySpec, GroupSpec]:
    """
    Recursively parse a Zarr group or Zarr array into an ArraySpec or GroupSpec.
    """

    if isinstance(element, zarr.Array):
        result = ArraySpec.from_zarr(element)
    elif isinstance(element, zarr.Group):
        items = {}
        for name, item in element.items():
            if isinstance(item, zarr.Array):
                _item = ArraySpec.from_zarr(item)
            elif isinstance(item, zarr.Group):
                _item = GroupSpec.from_zarr(item)
            items[name] = _item

        result = GroupSpec(attrs=dict(element.attrs), items=items)
        return result
    else:
        msg = f"""
        Object of type {type(element)} cannot be processed by this function. 
        This function can only parse objects that comply with the ArrayLike or 
        GroupLike protocols.
        """
        raise ValueError(msg)
    return result


def from_spec(
    store: BaseStore, path: str, spec: Union[ArraySpec, GroupSpec]
) -> Union[zarr.Array, zarr.Group]:
    """
    Materialize a zarr hierarchy on a given storage backend from an ArraySpec or
    GroupSpec.
    """
    if isinstance(spec, ArraySpec):
        result = spec.to_zarr(store, path)
    elif isinstance(spec, GroupSpec):
        result = spec.to_zarr(store, path)
    else:
        msg = f"""
        Invalid argument for spec. Expected an instance of GroupSpec or ArraySpec, got
        {type(spec)} instead.
        """
        raise ValueError(msg)

    return result
