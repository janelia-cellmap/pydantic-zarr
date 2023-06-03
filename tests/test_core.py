from pydantic import ValidationError
import pytest
import zarr
from zarr.errors import ContainsGroupError
from typing import Any, TypedDict
import numcodecs
from pydantic_zarr.core import ArraySpec, GroupSpec, to_zarr, from_zarr


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
    assert spec.attrs == array2.attrs
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

    spec = GroupSpec[RootAttrs, Any](
        attrs=RootAttrs(foo=10, bar=[0, 1, 2]),
        items={
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
    group = to_zarr(spec, store, "/group_a")

    # parse the spec from that group
    observed = from_zarr(group)
    assert observed == spec

    # materialize again
    with pytest.raises(ContainsGroupError):
        group = to_zarr(spec, store, "/group_a")

    group2 = to_zarr(spec, store, "/group_a", overwrite=True)
    assert group2 == group


def test_shape_chunks():
    for a, b in zip(range(1, 5), range(2, 6)):
        with pytest.raises(ValidationError):
            ArraySpec(shape=(1,) * a, chunks=(1,) * b, dtype="uint8", attrs={})
        with pytest.raises(ValidationError):
            ArraySpec(shape=(1,) * b, chunks=(1,) * a, dtype="uint8", attrs={})
