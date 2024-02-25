from __future__ import annotations

from typing import (
    Any,
    Dict,
    Generic,
    Literal,
    Mapping,
    TypeVar,
    Union,
    overload,
)
from typing_extensions import Annotated
from pydantic import AfterValidator, model_validator
from pydantic.functional_validators import BeforeValidator
from zarr.storage import init_group, BaseStore, contains_group
import numcodecs
import zarr
import os
import numpy as np
import numpy.typing as npt
from numcodecs.abc import Codec
from zarr.errors import ContainsGroupError
from pydantic_zarr.core import (
    IncEx,
    StrictBase,
    ensure_key_no_path,
    model_like,
)

TAttr = TypeVar("TAttr", bound=Mapping[str, Any])
TItem = TypeVar("TItem", bound=Union["GroupSpec", "ArraySpec"])


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


def parse_dimension_separator(data: Any) -> Literal["/", "."]:
    if data is None:
        return "/"
    if data in ("/", "."):
        return data
    raise ValueError(f'Invalid data, expected one of ("/", ".", None), got {data}')


CodecDict = Annotated[dict[str, Any], BeforeValidator(dictify_codec)]


class NodeSpec(StrictBase):
    """
    The base class for V2 ArraySpec and GroupSpec.

    Attributes
    ----------

    zarr_format: Literal[2]
        The Zarr version represented by this node. Must be 2.
    """

    zarr_version: Literal[2] = 2


class ArraySpec(NodeSpec, Generic[TAttr]):
    """
    A model of a Zarr Version 2 Array.

    Attributes
    ----------

    attributes: TAttr
        User-defined metadata associated with this array.
    shape: Sequence[int]
        The shape of this array.
    dtype: str
        The data type of this array.
    chunks: Sequence[int]
        The chunk size for this array.
    order: Union["C", "F"]
        The memory order of this array. Must be either "C", which designates "C order",
        or "F", which designates "F order".
    fill_value: FillValue
        The fill value for this array. The default is 0.
    compressor: Optional[CodecDict]
        A JSON-serializable representation of a compression codec, or None.
    dimension_separator: Union[".", "/"]
        The character used for separating chunk keys for this array. Must be either "/"
        or ".". The default is "/".
    """

    attributes: TAttr
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: DtypeStr
    fill_value: int | float | None = 0
    order: Literal["C", "F"] = "C"
    filters: list[CodecDict] | None = None
    dimension_separator: Annotated[
        Literal["/", "."], BeforeValidator(parse_dimension_separator)
    ] = "/"
    compressor: CodecDict | None = None

    @model_validator(mode="after")
    def check_ndim(self):
        """
        Check that the length of shape and chunks matches.
        """
        if (lshape := len(self.shape)) != (lchunks := len(self.chunks)):
            msg = (
                f"Length of shape must match length of chunks. Got {lshape} elements",
                f"for shape and {lchunks} elements for chunks.",
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
        self,
        store: BaseStore,
        path: str,
        **kwargs,
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
        result = zarr.create(store=store, path=path, **spec_dict, **kwargs)
        result.attrs.put(attrs)
        return result

    def like(
        self,
        other: ArraySpec | zarr.Array,
        include: IncEx = None,
        exclude: IncEx = None,
    ):
        """
        Compare a GroupSpec to another GroupSpec or a Zarr group, parameterized over the fields
        to exclude or include in the comparison. Models are first converted to dict via the
        `model_dump` method of pydantic.BaseModel, then compared with the `==` operator.

        Parameters
        ----------

        other: ArraySpec | zarr.Array
            The array (model or actual) to compare with. If other is a zarr.Array, it will be
            converted to ArraySpec first.
        include: IncEx
            A specification of fields to include in the comparison. Default value is `None`,
            which means that all fields will be included. See the documentation of
            `pydantic.BaseModel.model_dump` for more details.
        exclude: IncEx
            A specification of fields to exclude from the comparison. Default value is `None`,
            which means that no fields will be excluded. See the documentation of
            `pydantic.BaseModel.model_dump` for more details.

        Returns
        -------

        bool
            `True` if the two models have identical fields, `False` otherwise.
        """

        other_parsed: ArraySpec
        if isinstance(other, zarr.Array):
            other_parsed = ArraySpec.from_zarr(other)
        else:
            other_parsed = other

        result = model_like(self, other_parsed, include=include, exclude=exclude)
        return result


class GroupSpec(NodeSpec, Generic[TAttr, TItem]):
    """
    A model of a Zarr Version 2 Group.

    Attributes
    ----------

    attributes: TAttr
        The user-defined attributes of this group.
    members: dict[str, TItem] | None
        The members of this group. `members` is a dict with string keys and values that
        must inherit from either ArraySpec or GroupSpec.
    """

    attributes: TAttr
    members: Annotated[dict[str, TItem] | None, AfterValidator(ensure_key_no_path)] = {}

    @classmethod
    def from_zarr(cls, group: zarr.Group, depth: int = -1) -> "GroupSpec[TAttr, TItem]":
        """
        Create a GroupSpec from a Zarr group. Subgroups and arrays contained in the Zarr
        group will be converted to instances of GroupSpec and ArraySpec, respectively,
        and these spec instances will be stored in the .items attribute of the parent
        GroupSpec.

        Parameters
        ----------
        group : Zarr group

        depth: int
            An integer which may be no lower than -1. Determines how far into the tree to parse.

        Returns
        -------
        An instance of GroupSpec that represents the structure of the Zarr hierarchy.
        """

        result: GroupSpec[TAttr, TItem]
        attributes = group.attrs.asdict()
        members = {}

        if depth < -1:
            msg = (
                f"Invalid value for depth. Got {depth}, expected an integer "
                "greater than or equal to -1."
            )
            raise ValueError(msg)
        if depth == 0:
            return cls(attributes=attributes, members=None)
        new_depth = max(depth - 1, -1)
        for name, item in group.items():
            if isinstance(item, zarr.Array):
                # convert to dict before the final typed GroupSpec construction
                item_out = ArraySpec.from_zarr(item).model_dump()
            elif isinstance(item, zarr.Group):
                # convert to dict before the final typed GroupSpec construction
                item_out = GroupSpec.from_zarr(item, depth=new_depth).model_dump()
            else:
                msg = (
                    f"Unparseable object encountered: {type(item)}. Expected zarr.Array"
                    " or zarr.Group."
                )

                raise ValueError(msg)
            members[name] = item_out

        result = cls(attributes=attributes, members=members)
        return result

    def to_zarr(self, store: BaseStore, path: str, **kwargs):
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
        spec_dict = self.model_dump(exclude={"members": True})
        attrs = spec_dict.pop("attributes")

        if contains_group(store, path):
            if not kwargs.get("overwrite", False):
                msg = (
                    f"A group already exists at path {path}. "
                    "Call to_zarr with overwrite=True to delete the existing group."
                )
                raise ContainsGroupError(msg)
            else:
                init_group(store=store, overwrite=kwargs["overwrite"], path=path)

        result = zarr.group(store=store, path=path, **kwargs)
        result.attrs.put(attrs)
        if self.members is not None:
            for name, member in self.members.items():
                subpath = os.path.join(path, name)
                member.to_zarr(store, subpath, **kwargs)

        return result

    def like(
        self,
        other: GroupSpec | zarr.Group,
        include: IncEx = None,
        exclude: IncEx = None,
    ):
        """
        Compare a GroupSpec to another GroupSpec or a Zarr group, parameterized over the fields
        to exclude or include in the comparison. Models are first converted to dict via the
        `model_dump` method of pydantic.BaseModel, then compared with the `==` operator.

        Parameters
        ----------

        other: GroupSpec | zarr.Group
            The group (model or actual) to compare with. If other is a zarr.Group, it will be
            converted to GroupSpec first.
        include: IncEx
            A specification of fields to include in the comparison. Default value is `None`,
            which means that all fields will be included. See the documentation of
            `pydantic.BaseModel.model_dump` for more details.
        exclude: IncEx
            A specification of fields to exclude from the comparison. Default value is `None`,
            which means that no fields will be excluded. See the documentation of
            `pydantic.BaseModel.model_dump` for more details.

        Returns
        -------

        bool
            `True` if the two models have identical fields, `False` otherwise.
        """

        other_parsed: GroupSpec
        if isinstance(other, zarr.Group):
            other_parsed = GroupSpec.from_zarr(other)
        else:
            other_parsed = other

        result = model_like(self, other_parsed, include=include, exclude=exclude)
        return result

    @classmethod
    def from_flat(cls, data: Dict[str, ArraySpec | GroupSpec]):
        """
        Create a GroupSpec from a "flat" representation
        (a dict mapping string paths to ArraySpec or GroupSpec instances)
        """
        unflattened = unflatten_group(data)
        return cls(**unflattened.model_dump())


@overload
def from_zarr(element: zarr.Group) -> GroupSpec:
    ...


@overload
def from_zarr(element: zarr.Array) -> ArraySpec:
    ...


def from_zarr(
    element: zarr.Array | zarr.Group, depth: int = -1
) -> ArraySpec | GroupSpec:
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
        return result
    elif isinstance(element, zarr.Group):
        result = GroupSpec.from_zarr(element, depth=depth)
        return result
    msg = (
        f"Object of type {type(element)} cannot be processed by this function. "
        "This function can only parse zarr.Group or zarr.Array"
    )
    raise TypeError(msg)


@overload
def to_zarr(
    spec: ArraySpec,
    store: BaseStore,
    path: str,
    **kwargs,
) -> zarr.Array:
    ...


@overload
def to_zarr(
    spec: GroupSpec,
    store: BaseStore,
    path: str,
    **kwargs,
) -> zarr.Group:
    ...


def to_zarr(
    spec: ArraySpec | GroupSpec,
    store: BaseStore,
    path: str,
    **kwargs,
) -> zarr.Array | zarr.Group:
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
    A zarr Group or Array that is structurally equivalent to the spec object.
    This operation will create metadata documents in the store.

    """
    if isinstance(spec, ArraySpec):
        result = spec.to_zarr(store, path, **kwargs)
    elif isinstance(spec, GroupSpec):
        result = spec.to_zarr(store, path, **kwargs)
    else:
        msg = (
            "Invalid argument for spec. Expected an instance of GroupSpec or ",
            f"ArraySpec, got {type(spec)} instead.",
        )
        raise ValueError(msg)

    return result


def flatten(
    node: ArraySpec | GroupSpec, path: str = "/"
) -> Dict[str, ArraySpec | GroupSpec]:
    """
    Flatten a `GroupSpec` or `ArraySpec`.
    Takes a `GroupSpec` or `ArraySpec` and a string, and returns dictionary with string keys and values that are
    `GroupSpec` or `ArraySpec`. If the input is an `ArraySpec`, then this function just returns the dict `{path: node}`.
    If the input is a `GroupSpec`, then the resulting dictionary will contain a copy of the input with an empty `members` attribute
    under the key `path`, as well as copies of the result of calling `flatten_node` on each member of the input, each under a key created by joining `path` with a '/` character
    to the name of each member.

    Paramters
    ---------
    node: `GroupSpec` | `ArraySpec`
        The node to flatten.
    path: `str`, default is 'root'
        The root path. If the input is a `GroupSpec`, then the keys in `GroupSpec.members` will be
        made relative to `path` when used as keys in the result dictionary.

    Returns
    -------
    `Dict[str, GroupSpec | ArraySpec]`

    """
    result = {}
    model_copy: Union[ArraySpec, GroupSpec]
    if isinstance(node, ArraySpec):
        model_copy = node.model_copy(deep=True)
    else:
        model_copy = node.model_copy(deep=True, update={"members": None})
        if node.members is not None:
            for name, value in node.members.items():
                result.update(flatten(value, os.path.join(path, name)))

    result[path] = model_copy

    return result


def unflatten(tree: Dict[str, ArraySpec | GroupSpec]) -> ArraySpec | GroupSpec:
    """
    Wraps unflatten_group, handling the special case where a zarr array is defined at the root of
    a hierarchy and thus is not contained by a group.
    """

    if (
        len(tree.keys()) == 1
        and tuple(tree.keys()) == ("/",)
        and isinstance(tuple(tree.values())[0], ArraySpec)
    ):
        return tuple(tree.values())[0]
    else:
        return unflatten_group(tree)


def unflatten_group(data: Dict[str, ArraySpec | GroupSpec]) -> GroupSpec:
    """
    Generate a GroupSpec from a "flat" representation of a hierarchy, i.e. a dictionary with
    string keys (paths) and ArraySpec / GroupSpec values (nodes).
    """
    root_name = ""
    sep = "/"
    # arrays that will be members of the returned GroupSpec
    member_arrays: Dict[str, ArraySpec] = {}
    # groups, and their members, that will be members of the returned GroupSpec.
    # this dict is constructed by recursively applying this function.
    member_groups: Dict[str, GroupSpec] = {}
    # this dict collects the arrays and groups that belong to one of the members of the group
    # we are constructing. They will later be aggregated in a recursive step that populates
    # member_groups
    submember_by_parent_name: Dict[str, Dict[str, ArraySpec | GroupSpec]] = {}

    # Get the root node
    try:
        # The root node is a GroupSpec with the key "/"
        root_node = data.pop(root_name + sep)
        if isinstance(root_node, ArraySpec):
            raise ValueError("Got an ArraySpec as the root node. This is invalid.")
    except KeyError:
        # If a root node was not found, create a default one
        root_node = GroupSpec(attributes={}, members=None)

    # partition the tree (sans root node) into 2 categories: (arrays, groups + their members).
    for key, value in data.items():
        key_parts = key.split(sep)
        if key_parts[0] != root_name:
            raise ValueError(f'Invalid path: {key} does not start with "{sep}".')

        subparent_name = key_parts[1]

        if len(key_parts) == 2:
            # this is an array or group that belongs to the group we are ultimately returning
            if isinstance(value, ArraySpec):
                member_arrays[subparent_name] = value
            else:
                if subparent_name not in submember_by_parent_name:
                    submember_by_parent_name[subparent_name] = {}
                submember_by_parent_name[subparent_name][sep] = value
        else:
            # these are groups or arrays that belong to one of the member groups
            # not great that we repeat this conditional dict initialization
            if subparent_name not in submember_by_parent_name:
                submember_by_parent_name[subparent_name] = {}
            submember_by_parent_name[subparent_name][
                sep.join([root_name, *key_parts[2:]])
            ] = value

    # recurse
    for subparent_name, submemb in submember_by_parent_name.items():
        member_groups[subparent_name] = unflatten_group(submemb)

    return GroupSpec(
        members={**member_groups, **member_arrays}, attributes=root_node.attributes
    )
