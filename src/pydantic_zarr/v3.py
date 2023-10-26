from __future__ import annotations

from typing import (
    Any,
    Generic,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)
from pydantic import BaseModel, Field, ConfigDict
from zarr.storage import BaseStore
import zarr
import numpy as np
import numpy.typing as npt
from numcodecs.abc import Codec

TAttr = TypeVar("TAttr", bound=Union[Mapping[str, Any], BaseModel])
TItem = TypeVar("TItem", bound=Union["GroupSpec", "ArraySpec"])
TConfig = TypeVar("TConfig", bound=Union[Mapping[str, Any], BaseModel])
DimensionSeparator = Union[Literal["."], Literal["/"]]
ZarrVersion = Literal[3]
ArrayOrder = Union[Literal["C"], Literal["F"]]
NodeType = Union[Literal["group"], Literal["array"]]

# todo: introduce a type that represents hexadecimal representations of floats
FillValue = Union[
    bool,
    int,
    float,
    Literal["Infinity"],
    Literal["-Infinity"],
    Literal["NaN"],
    str,
    Tuple[float, float],
    Tuple[int, ...],
]


class StrictBase(BaseModel):
    model_config = ConfigDict(frozen=True)


class NamedConfig(StrictBase):
    name: str
    configuration: Optional[Union[Mapping[str, Any], BaseModel]]


class RegularChunkingConfig(StrictBase):
    chunk_shape: List[int]


class RegularChunking(NamedConfig):
    name: Literal["regular"] = "regular"
    config: RegularChunkingConfig


class DefaultChunkKeyEncodingConfig(StrictBase):
    separator: DimensionSeparator


class DefaultChunkKeyEncoding(NamedConfig):
    name: Literal["default"]
    config: Optional[DefaultChunkKeyEncodingConfig]


class NodeSpecV3(BaseModel, Generic[TAttr]):
    """
    The base class for ArraySpec and GroupSpec. Generic with respect to the type of its
    attributes.
    """

    zarr_format: ZarrVersion = 3
    node_type: NodeType

    class Config:
        extra = "forbid"


class ArraySpec(NodeSpecV3, Generic[TAttr]):
    """
    This pydantic model represents the structural properties of a zarr array.
    It does not represent the data contained in the array. It is generic with respect to
    the type of attributes.
    """

    node_type: NodeType = Field("array", frozen=True)
    attributes: Optional[TAttr]
    shape: tuple[int, ...]
    data_type: str
    chunk_grid: NamedConfig
    chunk_key_encoding: NamedConfig
    fill_value: FillValue  # todo: validate this against the data type
    codecs: List[NamedConfig]
    storage_transformers: List[NamedConfig]
    dimension_names: Optional[List[str]]  # todo: validate this against shape

    def stringify_dtype(cls, v):
        """
        Convert a np.dtype object into a string
        """
        return np.dtype(v).str

    def jsonify_compressor(cls, v):
        if isinstance(v, Codec):
            v = v.get_config()
        return v

    def jsonify_filters(cls, v):
        if v is not None:
            try:
                v = [element.get_config() for element in v]
                return v
            except AttributeError:
                pass
        return v

    def check_ndim(cls, values):
        if "shape" in values and "chunks" in values:
            if (lshape := len(values["shape"])) != (lchunks := len(values["chunks"])):
                msg = (
                    f"Length of shape must match length of chunks. Got {lshape} "
                    f"elements for shape and {lchunks} elements for chunks."
                )
                raise ValueError(msg)
        return values

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
        raise NotImplementedError

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
        raise NotImplementedError


class GroupSpec(NodeSpecV3, Generic[TAttr, TItem]):
    node_type: NodeType = Field("group", frozen=True)
    attributes: TAttr
    members: dict[str, TItem] = {}

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

        raise NotImplementedError

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
        raise NotImplementedError


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

    raise NotImplementedError


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
