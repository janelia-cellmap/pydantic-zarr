from __future__ import annotations

from typing import (
    Any,
    Dict,
    Generic,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
    overload,
)
from typing_extensions import Annotated
from pydantic import BaseModel, model_validator
from pydantic.functional_validators import AfterValidator, BeforeValidator
from zarr.storage import init_group, BaseStore
import numcodecs
import zarr
import os
import numpy as np
import numpy.typing as npt
from numcodecs.abc import Codec

from pydantic_zarr.core import StrictBase

TAttr = TypeVar("TAttr", bound=Union[Mapping[str, Any], BaseModel])
TItem = TypeVar("TItem", bound=Union["ArraySpec", "GroupSpec"])

DimensionSeparator = Union[Literal["."], Literal["/"]]
ZarrVersion = Union[Literal[2], Literal[3]]
ArrayOrder = Union[Literal["C"], Literal["F"]]


def stringify_dtype(v: npt.DTypeLike):
    """
    Convert a np.dtype-like object into a string
    """
    return np.dtype(v).str


DtypeStr = Annotated[str, AfterValidator(stringify_dtype)]


def dictify_codec(v: Any) -> Any:
    """
    Convert a numcodecs codec to json-compatible dict
    """
    if isinstance(v, Codec):
        v = v.get_config()
    return v


CodecLike = Annotated[Dict[str, Any], BeforeValidator(dictify_codec)]


class NodeSpecV2(StrictBase, Generic[TAttr]):
    """
    The base class for ArraySpec and GroupSpec.
    """

    zarr_version: ZarrVersion = 2


class ArraySpec(NodeSpecV2, Generic[TAttr]):
    """
    This pydantic model represents the structural properties of a zarr array.
    It does not represent the data contained in the array. It is generic with respect to
    the type of attributes.
    """

    attributes: TAttr
    shape: Sequence[int]
    chunks: Sequence[int]
    dtype: DtypeStr
    fill_value: Union[None, int, float] = 0
    order: ArrayOrder = "C"
    filters: Optional[list[CodecLike]] = None
    dimension_separator: DimensionSeparator = "/"
    compressor: Optional[CodecLike] = None

    @model_validator(mode="after")
    def check_ndim(self) -> "ArraySpec":
        if len(self.shape) != len(self.chunks):
            msg = (
                f"Length of shape must match length of chunks. Got {len(self.shape)} "
                f"elements for shape, but {len(self.chunks)} elements for chunks."
            )
            raise ValueError(msg)
        return self

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
            fill_value=zarray.fill_value,
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
        attributes = spec_dict.pop("attributes")
        if self.compressor is not None:
            spec_dict["compressor"] = numcodecs.get_codec(spec_dict["compressor"])
        if self.filters is not None:
            spec_dict["filters"] = [
                numcodecs.get_codec(f) for f in spec_dict["filters"]
            ]
        result = zarr.create(store=store, path=path, **spec_dict, overwrite=overwrite)
        result.attrs.put(attributes)
        return result


class GroupSpec(NodeSpecV2, Generic[TAttr, TItem]):
    attributes: TAttr
    members: Mapping[str, TItem] = {}

    # @field_validator('members')
    # def validate_members():

    @classmethod
    def from_zarr(cls, group: zarr.Group) -> "GroupSpec[TAttr, TItem]":
        """
        Create a GroupSpec from a zarr group. Subgroups and arrays contained in the zarr
        group will be converted to instances of GroupSpec and ArraySpec, respectively,
        and these spec instances will be stored in the .members attribute of the parent
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
        members = {}
        for name, member in group.items():
            if isinstance(member, zarr.Array):
                _item = ArraySpec.from_zarr(member)
            elif isinstance(member, zarr.Group):
                _item = GroupSpec.from_zarr(member)
            else:
                msg = (
                    f"Unparseable object encountered: {type(member)}. "
                    "Expected zarr.Array or zarr.Group.",
                )
                raise ValueError(msg)
            members[name] = _item
        result = cls(attributes=group.attrs.asdict(), members=members)
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
        # pop members because it's not a valid kwarg for init_group
        spec_dict.pop("members")
        # pop attributes because it's not a valid kwarg for init_group
        attributes = spec_dict.pop("attributes")
        # weird that we have to call init_group before creating the group
        init_group(store, overwrite=overwrite, path=path)
        result = zarr.group(store=store, path=path, **spec_dict, overwrite=overwrite)
        result.attrs.put(attributes)
        for name, member in self.members.items():
            subpath = os.path.join(path, name)
            member.to_zarr(store, subpath, overwrite=overwrite)

        return result


@overload
def from_zarr(element: zarr.Array) -> ArraySpec:
    ...


@overload
def from_zarr(element: zarr.Group) -> GroupSpec:
    ...


def from_zarr(element: Union[zarr.Array, zarr.Group]) -> Union[ArraySpec, GroupSpec]:
    """
    Recursively parse a Zarr group or Zarr array into an ArraySpec or GroupSpec.

    Parameters
    ---------
    element : a zarr Array or zarr Group

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
                msg = (f"Unparseable object encountered: {type(member)}.",)
                "Expected zarr.Array or zarr.Group."
                raise ValueError(msg)
            members[name] = _item

        result = GroupSpec(attributes=element.attrs.asdict(), members=members)
        return result
    else:
        msg = (
            f"Object of type {type(element)} cannot be processed by this function. ",
            "This function can only parse objects that comply with the ArrayLike or ",
            "GroupLike protocols.",
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
        msg = ("Invalid argument for spec. Expected an instance of GroupSpec or ",)
        f"ArraySpec, got {type(spec)} instead."
        raise ValueError(msg)

    return result
