from __future__ import annotations

from typing import (
    Any,
    Generic,
    Iterable,
    Literal,
    Optional,
    TypeVar,
    Union,
    Protocol,
    runtime_checkable,
)

from pydantic.generics import GenericModel
from zarr.storage import init_group, BaseStore
import zarr
import os

TAttrs = TypeVar("TAttrs", bound=dict[str, Any])

DimensionSeparator = Union[Literal["."], Literal["/"]]
ZarrVersion = Union[Literal[2], Literal[3]]
ArrayOrder = Union[Literal["C"], Literal["F"]]


class NodeSpec(GenericModel, Generic[TAttrs]):
    zarr_version: ZarrVersion = 2
    attrs: TAttrs = {}


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
    filters: dict[str, Any] = {}
    dimension_separator: DimensionSeparator = "/"
    compressor: Optional[dict[str, Any]] = None

    @classmethod
    def from_array(
        cls,
        data: ArrayLike,
        chunks,
        fill_value=0,
        order="C",
        filters={},
        dimension_separator="/",
        compressor=None,
        attrs={},
    ):
        """
        Generate an ArraySpec from an arraylike object
        """
        return cls(
            shape=data.shape,
            dtype=data.dtype,
            chunks=chunks,
            fill_value=fill_value,
            order=order,
            filters=filters,
            dimension_separator=dimension_separator,
            compressor=compressor,
            attrs=attrs,
        )


TChild = TypeVar("TChild", bound=Union["ArraySpec", "GroupSpec"])


class GroupSpec(NodeSpec, Generic[TAttrs, TChild]):
    children: dict[str, TChild] = {}


@runtime_checkable
class NodeLike(Protocol):
    basename: str
    attrs: dict[str, Any]


@runtime_checkable
class ArrayLike(NodeLike, Protocol):
    attrs: dict[str, Any]
    fill_value: Any
    chunks: tuple[int, ...]
    shape: tuple[int, ...]
    dtype: str


@runtime_checkable
class GroupLike(NodeLike, Protocol):
    def values(self) -> Iterable[Union[GroupLike, ArrayLike]]:
        """
        Iterable of the children of this group
        """
        ...


def to_spec(element: NodeLike) -> tuple(str, Union[ArraySpec, GroupSpec]):
    """
    Recursively parse a Zarr group or Zarr array into an ArraySpec or GroupSpec.
    """

    if isinstance(element, ArrayLike):
        result = (
            element.basename,
            ArraySpec(
                shape=element.shape,
                dtype=str(element.dtype),
                attrs=dict(element.attrs),
                chunks=element.chunks,
                fill_value=element.fill_value,
            ),
        )
    elif isinstance(element, GroupLike):
        children = tuple(map(to_spec, element.values()))
        result = (
            element.basename,
            GroupSpec(attrs=dict(element.attrs), children=children),
        )
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
        spec_dict = spec.dict()
        attrs = spec_dict.pop("attrs")
        result: zarr.Array = zarr.create(store=store, path=path, **spec_dict)
        result.attrs.put(attrs)

    elif isinstance(spec, GroupSpec):
        spec_dict = spec.dict()
        spec_dict.pop("children")
        attrs = spec_dict.pop("attrs")
        # needing to call init_group, then zarr.group is not ergonomic
        init_group(store=store, path=path)
        result = zarr.group(store=store, path=path, **spec_dict)
        result.attrs.put(attrs)
        for name, child in spec.children.items():
            subpath = os.path.join(path, name)
            from_spec(store, subpath, child)
    else:
        msg = f"""
        Invalid argument for spec. Expected an instance of GroupSpec or ArraySpec, got
        {type(spec)} instead.
        """
        raise ValueError(msg)

    return result
