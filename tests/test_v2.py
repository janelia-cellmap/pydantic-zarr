from pydantic import ValidationError
import pytest
import zarr
from zarr.errors import ContainsGroupError
from typing import Any, Literal, Union, Optional
import numcodecs
from numcodecs.abc import Codec
from pydantic_zarr.v2 import ArraySpec, GroupSpec, to_zarr, from_zarr
import numpy as np
import numpy.typing as npt
import sys

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


@pytest.mark.parametrize("chunks", ((1,), (1, 2), ((1, 2, 3))))
@pytest.mark.parametrize("order", ("C", "F"))
@pytest.mark.parametrize("dtype", ("bool", "uint8", "float64"))
@pytest.mark.parametrize("dimension_separator", (".", "/"))
@pytest.mark.parametrize("compressor", (None, numcodecs.LZMA(), numcodecs.GZip()))
@pytest.mark.parametrize(
    "filters", (None, ("delta",), ("scale_offset",), ("delta", "scale_offset"))
)
def test_array_spec(
    chunks: tuple[int, ...],
    order: str,
    dtype: str,
    dimension_separator: str,
    compressor: Optional[Codec],
    filters: Optional[tuple[str, ...]],
):
    store = zarr.MemoryStore()
    _filters: Optional[list[Codec]]
    if filters is not None:
        _filters = []
        for filter in filters:
            if filter == "delta":
                _filters.append(numcodecs.Delta(dtype))
            if filter == "scale_offset":
                _filters.append(numcodecs.FixedScaleOffset(0, 1.0, dtype=dtype))
    else:
        _filters = filters

    array = zarr.create(
        (100,) * len(chunks),
        path="foo",
        store=store,
        chunks=chunks,
        dtype=dtype,
        order=order,
        dimension_separator=dimension_separator,
        compressor=compressor,
        filters=_filters,
    )
    array.attrs.put({"foo": [100, 200, 300], "bar": "hello"})
    spec = ArraySpec.from_zarr(array)

    assert spec.zarr_version == array._version
    assert spec.dtype == array.dtype
    assert spec.attributes == array.attrs
    assert spec.chunks == array.chunks

    assert spec.dimension_separator == array._dimension_separator
    assert spec.shape == array.shape
    assert spec.fill_value == array.fill_value
    # this is a sign that nullability is being misused in zarr-python
    # the correct approach would be to use an empty list to express "no filters".
    if array.filters is not None:
        assert spec.filters == [f.get_config() for f in array.filters]
    else:
        assert spec.filters == array.filters

    if array.compressor is not None:
        assert spec.compressor == array.compressor.get_config()
    else:
        assert spec.compressor == array.compressor

    assert spec.order == array.order

    array2 = spec.to_zarr(store, "foo2")

    assert spec.zarr_version == array2._version
    assert spec.dtype == array2.dtype
    assert spec.attributes == array2.attrs
    assert spec.chunks == array2.chunks

    if array2.compressor is not None:
        assert spec.compressor == array2.compressor.get_config()
    else:
        assert spec.compressor == array2.compressor

    if array2.filters is not None:
        assert spec.filters == [f.get_config() for f in array2.filters]
    else:
        assert spec.filters == array2.filters

    assert spec.dimension_separator == array2._dimension_separator
    assert spec.shape == array2.shape
    assert spec.fill_value == array2.fill_value


@pytest.mark.parametrize("array", (np.arange(10), np.zeros((10, 10), dtype="uint8")))
def test_array_spec_from_array(array: npt.NDArray[Any]):
    spec = ArraySpec.from_array(array)
    assert spec.dtype == array.dtype.str
    assert np.dtype(spec.dtype) == array.dtype
    assert spec.shape == array.shape
    assert spec.chunks == array.shape
    assert spec.attributes == {}

    attrs = {"foo": 10}
    chunks = (1,) * array.ndim
    spec2 = ArraySpec.from_array(array, attributes=attrs, chunks=chunks)
    assert spec2.chunks == chunks
    assert spec2.attributes == attrs
    assert spec.dtype == array.dtype.str
    assert np.dtype(spec.dtype) == array.dtype
    assert spec.shape == array.shape


@pytest.mark.parametrize("chunks", ((1,), (1, 2), ((1, 2, 3))))
@pytest.mark.parametrize("order", ("C", "F"))
@pytest.mark.parametrize("dtype", ("bool", "uint8", np.dtype("uint8"), "float64"))
@pytest.mark.parametrize("dimension_separator", (".", "/"))
@pytest.mark.parametrize(
    "compressor", (numcodecs.LZMA().get_config(), numcodecs.GZip())
)
@pytest.mark.parametrize(
    "filters", (None, ("delta",), ("scale_offset",), ("delta", "scale_offset"))
)
def test_serialize_deserialize_groupspec(
    chunks: tuple[int, ...],
    order: Literal["C", "F"],
    dtype: str,
    dimension_separator: Literal[".", "/"],
    compressor: Any,
    filters: tuple[str, ...],
):
    _filters = filters
    if _filters is not None:
        _filters = []
        for filter in filters:
            if filter == "delta":
                _filters.append(numcodecs.Delta(dtype))
            if filter == "scale_offset":
                _filters.append(numcodecs.FixedScaleOffset(0, 1.0, dtype=dtype))

    class RootAttrs(TypedDict):
        foo: int
        bar: list[int]

    class SubGroupAttrs(TypedDict):
        a: str
        b: float

    SubGroup = GroupSpec[SubGroupAttrs, Any]

    class ArrayAttrs(TypedDict):
        scale: list[float]

    store = zarr.MemoryStore()

    spec = GroupSpec[RootAttrs, Union[ArraySpec, SubGroup]](
        attributes=RootAttrs(foo=10, bar=[0, 1, 2]),
        members={
            "s0": ArraySpec[ArrayAttrs](
                shape=(10,) * len(chunks),
                chunks=chunks,
                dtype=dtype,
                filters=_filters,
                compressor=compressor,
                order=order,
                dimension_separator=dimension_separator,
                attributes=ArrayAttrs(scale=[1.0]),
            ),
            "s1": ArraySpec[ArrayAttrs](
                shape=(5,) * len(chunks),
                chunks=chunks,
                dtype=dtype,
                filters=_filters,
                compressor=compressor,
                order=order,
                dimension_separator=dimension_separator,
                attributes=ArrayAttrs(scale=[2.0]),
            ),
            "subgroup": SubGroup(attributes=SubGroupAttrs(a="foo", b=1.0)),
        },
    )
    # check that the model round-trips dict representation
    assert spec == GroupSpec(**spec.model_dump())

    # materialize a zarr group, based on the spec
    group = to_zarr(spec, store, "/group_a")

    # parse the spec from that group
    observed = from_zarr(group)
    assert observed == spec

    # check that we can't overwrite the original group
    with pytest.raises(ContainsGroupError):
        group = to_zarr(spec, store, "/group_a")

    # materialize again with overwrite
    group2 = to_zarr(spec, store, "/group_a", overwrite=True)
    assert group2 == group

    # again with class methods
    group3 = spec.to_zarr(store, "/group_b")
    observed = spec.from_zarr(group3)
    assert observed == spec


def test_shape_chunks():
    """
    Test that the length of the chunks and the shape match
    """
    for a, b in zip(range(1, 5), range(2, 6)):
        with pytest.raises(ValidationError):
            ArraySpec(shape=(1,) * a, chunks=(1,) * b, dtype="uint8", attributes={})
        with pytest.raises(ValidationError):
            ArraySpec(shape=(1,) * b, chunks=(1,) * a, dtype="uint8", attributes={})


def test_validation() -> None:
    """
    Test that specialized GroupSpec and ArraySpec instances cannot be serialized from
    the wrong inputs without a ValidationError.
    """

    class GroupAttrsA(TypedDict):
        group_a: bool

    class GroupAttrsB(TypedDict):
        group_b: bool

    class ArrayAttrsA(TypedDict):
        array_a: bool

    class ArrayAttrsB(TypedDict):
        array_b: bool

    ArrayA = ArraySpec[ArrayAttrsA]
    ArrayB = ArraySpec[ArrayAttrsB]
    GroupA = GroupSpec[GroupAttrsA, ArrayA]
    GroupB = GroupSpec[GroupAttrsB, ArrayB]

    store = zarr.MemoryStore

    specA = GroupA(
        attributes=GroupAttrsA(group_a=True),
        members={
            "a": ArrayA(
                attributes=ArrayAttrsA(array_a=True),
                shape=(100,),
                dtype="uint8",
                chunks=(10,),
            )
        },
    )

    specB = GroupB(
        attributes=GroupAttrsB(group_b=True),
        members={
            "a": ArrayB(
                attributes=ArrayAttrsB(array_b=True),
                shape=(100,),
                dtype="uint8",
                chunks=(10,),
            )
        },
    )

    # check that we cannot create a specialized GroupSpec with the wrong attributes
    with pytest.raises(ValidationError):
        GroupB(
            attributes=GroupAttrsA(group_a=True),
            members={},
        )

    store = zarr.MemoryStore()
    groupAMat = specA.to_zarr(store, path="group_a")
    groupBMat = specB.to_zarr(store, path="group_b")

    GroupA.from_zarr(groupAMat)
    GroupB.from_zarr(groupBMat)
    ArrayA.from_zarr(groupAMat["a"])
    ArrayB.from_zarr(groupBMat["a"])

    with pytest.raises(ValidationError):
        ArrayA.from_zarr(groupBMat["a"])

    with pytest.raises(ValidationError):
        ArrayB.from_zarr(groupAMat["a"])

    with pytest.raises(ValidationError):
        GroupB.from_zarr(groupAMat)

    with pytest.raises(ValidationError):
        GroupA.from_zarr(groupBMat)


@pytest.mark.parametrize("shape", ((1,), (2, 2), (3, 4, 5)))
@pytest.mark.parametrize("dtype", (None, "uint8", "float32"))
def test_from_array(shape, dtype):
    template = np.zeros(shape=shape, dtype=dtype)
    spec = ArraySpec.from_array(template)

    assert spec.shape == template.shape
    assert np.dtype(spec.dtype) == np.dtype(template.dtype)
    assert spec.chunks == template.shape
    assert spec.attributes == {}

    chunks = template.ndim * (1,)
    attrs = {"foo": 100}
    spec2 = ArraySpec.from_array(template, chunks=chunks, attributes=attrs)
    assert spec2.chunks == chunks
    assert spec2.attributes == attrs
