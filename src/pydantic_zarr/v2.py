from __future__ import annotations

from typing import (
    Any,
    Generic,
    Literal,
    Mapping,
    Optional,
    TypeVar,
    Union,
    overload,
)
from typing_extensions import Annotated
from pydantic import BaseModel, model_validator
from pydantic.functional_validators import BeforeValidator
from zarr.storage import init_group, BaseStore
import numcodecs
import zarr
import os
import numpy as np
import numpy.typing as npt
from numcodecs.abc import Codec

from pydantic_zarr.core import StrictBase

TAttr = TypeVar("TAttr", bound=Union[Mapping[str, Any], BaseModel])
TItem = TypeVar("TItem", bound=Union["GroupSpec", "ArraySpec"])

DimensionSeparator = Union[Literal["."], Literal["/"]]
ZarrVersion = Union[Literal[2], Literal[3]]
ArrayOrder = Union[Literal["C"], Literal["F"]]


def stringify_dtype(value: npt.DTypeLike):
    """
    Convert a np.dtype object into a string

    Paramters
    ---------

    value: DTypeLike
        Some object that can be coerced to a numpy dtype

    Returns
    -------

    A numpy dtype string representation of the input.
    """
    return np.dtype(value).str


DtypeStr = Annotated[str, BeforeValidator(stringify_dtype)]


def dictify_codec(value: Union[dict[str, Any], Codec]) -> dict[str, Any]:
    """
    Ensure that a numcodecs `Codec` is converted to a dict. If the input is not an
    insance of `numcodecs.abc.Codec`, then it is assumed to be a dict with string keys
    and it is returned unaltered.

    Paramters
    ---------

    value : Union[dict[str, Any], numcodecs.abc.Codec]
        The value to be dictified if it is not already a dict.

    Returns
    -------

    If the input was a `Codec`, then the result of calling `get_config()` on that
    object is returned. This should be a dict with string keys. All other values pass
    through unaltered.
    """
    if isinstance(value, Codec):
        result = value.get_config()
    else:
        result = value
    return result


CodecDict = Annotated[dict[str, Any], BeforeValidator(dictify_codec)]


class ArraySpec(StrictBase, Generic[TAttr]):
    """
    This pydantic model represents the structural properties of a zarr array.
    It does not represent the data contained in the array. It is generic with respect to
    the type of attrs.
    """

    zarr_version: ZarrVersion = 2
    attributes: TAttr
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: DtypeStr
    fill_value: Union[None, int, float] = 0
    order: ArrayOrder = "C"
    filters: Optional[list[CodecDict]] = None
    dimension_separator: DimensionSeparator = "/"
    compressor: Optional[CodecDict] = None

    @model_validator(mode="after")
    def check_ndim(cls, value):
        """
        Check that the length of shape and chunks matches.
        """
        if (lshape := len(value.shape)) != (lchunks := len(value.chunks)):
            msg = (
                f"Length of shape must match length of chunks. Got {lshape} elements",
                f"for shape and {lchunks} elements for chunks.",
            )
            raise ValueError(msg)
        return value

    @classmethod
    def from_array(cls, array: npt.NDArray[Any], **kwargs):
        """
        Create an ArraySpec from a numpy array-like object.

        Parameters
        ----------
        array : object that conforms to the numpy array API.
            The shape and dtype of this object will be used to construct an ArraySpec.
            If the `chunks` keyword argument is not given, the shape of the array will
            be used for the chunks.

        **kwargs : keyword arguments to the ArraySpec class constructor.

        Returns
        -------
        An instance of ArraySpec with properties derived from the provided array.

        """
        return cls(
            shape=array.shape,
            dtype=str(array.dtype),
            chunks=kwargs.pop("chunks", array.shape),
            attributes=kwargs.pop("attributes", {}),
            **kwargs,
        )

    @classmethod
    def from_zarr(cls, zarray: zarr.Array):
        """
        Create an ArraySpec from a zarr array.

        Parameters
        ----------
        zarray : zarr array

        Returns
        -------
        An instance of ArraySpec with properties derived from the provided zarr
        array.

        """
        return cls(
            shape=zarray.shape,
            chunks=zarray.chunks,
            dtype=str(zarray.dtype),
            # explicitly cast to numpy type and back to python
            # so that int 0 isn't serialized as 0.0
            fill_value=zarray.dtype.type(zarray.fill_value).tolist(),
            order=zarray.order,
            filters=zarray.filters,
            dimension_separator=zarray._dimension_separator,
            compressor=zarray.compressor,
            attributes=zarray.attrs.asdict(),
        )

    def to_zarr(
        self, store: BaseStore, path: str, overwrite: bool = False
    ) -> zarr.Array:
        """
        Serialize an ArraySpec to a zarr array at a specific path in a zarr store.

        Parameters
        ----------
        store : instance of zarr.BaseStore
            The storage backend that will manifest the array.

        path : str
            The location of the array inside the store.

        overwrite : bool
            Whether to overwrite an existing array or group at the path. If overwrite is
            False and an array or group already exists at the path, an exception will be
            raised. Defaults to False.

        Returns
        -------
        A zarr array that is structurally identical to the ArraySpec.
        This operation will create metadata documents in the store.
        """
        spec_dict = self.model_dump()
        attrs = spec_dict.pop("attributes")
        if self.compressor is not None:
            spec_dict["compressor"] = numcodecs.get_codec(spec_dict["compressor"])
        if self.filters is not None:
            spec_dict["filters"] = [
                numcodecs.get_codec(f) for f in spec_dict["filters"]
            ]
        result = zarr.create(store=store, path=path, **spec_dict, overwrite=overwrite)
        result.attrs.put(attrs)
        return result


class GroupSpec(StrictBase, Generic[TAttr, TItem]):
    zarr_version: ZarrVersion = 2
    attributes: TAttr
    members: dict[str, TItem] = {}

    @classmethod
    def from_zarr(cls, group: zarr.Group) -> "GroupSpec[TAttr, TItem]":
        """
        Create a GroupSpec from a zarr group. Subgroups and arrays contained in the zarr
        group will be converted to instances of GroupSpec and ArraySpec, respectively,
        and these spec instances will be stored in the .items attribute of the parent
        GroupSpec. This occurs recursively, so the entire zarr hierarchy below a given
        group can be represented as a GroupSpec.

        Parameters
        ----------
        group : zarr group

        Returns
        -------
        An instance of GroupSpec that represents the structure of the zarr hierarchy.
        """

        result: GroupSpec[TAttr, TItem]
        items = {}
        for name, item in group.items():
            if isinstance(item, zarr.Array):
                # convert to dict before the final typed GroupSpec construction
                _item = ArraySpec.from_zarr(item).model_dump()
            elif isinstance(item, zarr.Group):
                # convert to dict before the final typed GroupSpec construction
                _item = GroupSpec.from_zarr(item).model_dump()
            else:
                msg = (
                    f"Unparseable object encountered: {type(item)}. Expected zarr.Array"
                    " or zarr.Group."
                )

                raise ValueError(msg)
            items[name] = _item

        result = cls(attributes=group.attrs.asdict(), members=items)
        return result

    def to_zarr(self, store: BaseStore, path: str, overwrite: bool = False):
        """
        Serialize a GroupSpec to a zarr group at a specific path in a zarr store.

        Parameters
        ----------
        store : instance of zarr.BaseStore
            The storage backend that will manifest the group and its contents.

        path : str
            The location of the group inside the store.

        overwrite : bool
            Whether to overwrite an existing array or group at the path. If overwrite is
            False and an array or group already exists at the path, an exception will be
            raised. Defaults to False.

        Returns
        -------
        A zarr group that is structurally identical to the GroupSpec.
        This operation will create metadata documents in the store.
        """
        spec_dict = self.model_dump()
        # pop items because it's not a valid kwarg for init_group
        spec_dict.pop("members")
        # pop attrs because it's not a valid kwarg for init_group
        attrs = spec_dict.pop("attributes")
        # weird that we have to call init_group before creating the group
        init_group(store, overwrite=overwrite, path=path)
        result = zarr.group(store=store, path=path, **spec_dict, overwrite=overwrite)
        result.attrs.put(attrs)
        for name, member in self.members.items():
            subpath = os.path.join(path, name)
            member.to_zarr(store, subpath, overwrite=overwrite)

        return result


@overload
def from_zarr(element: zarr.Group) -> GroupSpec:
    ...


@overload
def from_zarr(element: zarr.Array) -> ArraySpec:
    ...


def from_zarr(element: Union[zarr.Array, zarr.Group]) -> Union[ArraySpec, GroupSpec]:
    """
    Recursively parse a Zarr group or Zarr array into an untyped ArraySpec or GroupSpec.

    Parameters
    ---------
    element : Union[zarr.Array, zarr.Group]

    Returns
    -------
    An instance of GroupSpec or ArraySpec that represents the
    structure of the zarr group or array.
    """

    if isinstance(element, zarr.Array):
        result = ArraySpec.from_zarr(element)
    elif isinstance(element, zarr.Group):
        members = {}
        for name, member in element.items():
            if isinstance(member, zarr.Array):
                _item = ArraySpec.from_zarr(member)
            elif isinstance(member, zarr.Group):
                _item = GroupSpec.from_zarr(member)
            else:
                msg = (
                    f"Unparseable object encountered: {type(member)}. Expected "
                    "zarr.Array or zarr.Group.",
                )
                raise ValueError(msg)
            members[name] = _item

        result = GroupSpec(attributes=element.attrs.asdict(), members=members)
        return result
    else:
        msg = (
            f"Object of type {type(element)} cannot be processed by this function. "
            "This function can only parse zarr.Group or zarr.Array"
        )
        raise ValueError(msg)
    return result


@overload
def to_zarr(
    spec: ArraySpec,
    store: BaseStore,
    path: str,
    overwrite: bool = False,
) -> zarr.Array:
    ...


@overload
def to_zarr(
    spec: GroupSpec,
    store: BaseStore,
    path: str,
    overwrite: bool = False,
) -> zarr.Group:
    ...


def to_zarr(
    spec: Union[ArraySpec, GroupSpec],
    store: BaseStore,
    path: str,
    overwrite: bool = False,
) -> Union[zarr.Array, zarr.Group]:
    """
    Serialize a GroupSpec or ArraySpec to a zarr group or array at a specific path in
    a zarr store.

    Parameters
    ----------
    spec : GroupSpec or ArraySpec
        The GroupSpec or ArraySpec that will be serialized to storage.

    store : instance of zarr.BaseStore
        The storage backend that will manifest the group or array.

    path : str
        The location of the group or array inside the store.

    overwrite : bool
       Whether to overwrite an existing array or group at the path. If overwrite is
        False and an array or group already exists at the path, an exception will be
        raised. Defaults to False.

    Returns
    -------
    A zarr Group or Array that is structurally identical to the spec object.
    This operation will create metadata documents in the store.

    """
    if isinstance(spec, ArraySpec):
        result = spec.to_zarr(store, path, overwrite=overwrite)
    elif isinstance(spec, GroupSpec):
        result = spec.to_zarr(store, path, overwrite=overwrite)
    else:
        msg = (
            "Invalid argument for spec. Expected an instance of GroupSpec or ",
            f"ArraySpec, got {type(spec)} instead.",
        )
        raise ValueError(msg)

    return result
