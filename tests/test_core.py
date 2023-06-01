import pytest
import zarr
from typing import Any, TypedDict
import numcodecs
from pydantic_zarr.core import ArraySpec, GroupSpec, from_spec, to_spec


@pytest.mark.parametrize("chunks", ((1,), (1, 2), ((1, 2, 3))))
@pytest.mark.parametrize("order", ("C", "F"))
@pytest.mark.parametrize("dtype", ("bool", "uint8", "float64"))
@pytest.mark.parametrize("dimension_separator", (".", "/"))
@pytest.mark.parametrize("compressor", (numcodecs.LZMA(), numcodecs.GZip()))
@pytest.mark.parametrize(
    "filters", (None, ("delta",), ("scale_offset",), ("delta", "scale_offset"))
)
def test_array_spec(
    chunks: tuple[int, ...],
    order: str,
    dtype: str,
    dimension_separator: str,
    compressor: Any,
    filters: tuple[str, ...],
):
    store = zarr.MemoryStore()

    _filters = filters
    if _filters is not None:
        _filters = []

        for filter in filters:
            if filter == "delta":
                _filters.append(numcodecs.Delta(dtype))
            if filter == "scale_offset":
                _filters.append(numcodecs.FixedScaleOffset(0, 1.0, dtype=dtype))

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
    assert spec.attrs == array.attrs
    assert spec.chunks == array.chunks
    assert spec.compressor == array.compressor.get_config()
    assert spec.dimension_separator == array._dimension_separator
    assert spec.shape == array.shape
    assert spec.fill_value == array.fill_value
    # this is a sign that nullability is being misused in zarr-python
    # the correct approach would be to use an empty list to express "no filters".
    if array.filters is not None:
        assert spec.filters == [f.get_config() for f in array.filters]
    else:
        assert spec.filters == array.filters
    assert spec.order == array.order


@pytest.mark.parametrize("chunks", ((1,), (1, 2), ((1, 2, 3))))
@pytest.mark.parametrize("order", ("C", "F"))
@pytest.mark.parametrize("dtype", ("bool", "uint8", "float64"))
@pytest.mark.parametrize("dimension_separator", (".", "/"))
@pytest.mark.parametrize(
    "compressor", (numcodecs.LZMA().get_config(), numcodecs.GZip().get_config())
)
@pytest.mark.parametrize(
    "filters", (None, ("delta",), ("scale_offset",), ("delta", "scale_offset"))
)
def test_serde(
    chunks: tuple[int, ...],
    order: str,
    dtype: str,
    dimension_separator: str,
    compressor: Any,
    filters: tuple[str, ...],
):

    _filters = filters
    if _filters is not None:
        _filters = []
        for filter in filters:
            if filter == "delta":
                _filters.append(numcodecs.Delta(dtype).get_config())
            if filter == "scale_offset":
                _filters.append(
                    numcodecs.FixedScaleOffset(0, 1.0, dtype=dtype).get_config()
                )

    class RootAttrs(TypedDict):
        foo: int
        bar: list[int]

    class SubGroupAttrs(TypedDict):
        a: str
        b: float

    class ArrayAttrs(TypedDict):
        scale: list[float]

    store = zarr.MemoryStore()

    spec = GroupSpec(
        attrs=RootAttrs(foo=10, bar=[0, 1, 2]),
        children={
            "s0": ArraySpec(
                shape=(10,) * len(chunks),
                chunks=chunks,
                dtype=dtype,
                filters=_filters,
                compressor=compressor,
                order=order,
                dimension_separator=dimension_separator,
                attrs=ArrayAttrs(scale=[1.0]),
            ),
            "s1": ArraySpec(
                shape=(5,) * len(chunks),
                chunks=chunks,
                dtype=dtype,
                filters=_filters,
                compressor=compressor,
                order=order,
                dimension_separator=dimension_separator,
                attrs=ArrayAttrs(scale=[2.0]),
            ),
            "subgroup": GroupSpec(attrs=SubGroupAttrs(a="foo", b=1.0)),
        },
    )

    # materialize a zarr group, based on the spec
    group = from_spec(store, "/group_a", spec)

    # parse the spec from that group
    observed = to_spec(group)
    assert observed == spec
