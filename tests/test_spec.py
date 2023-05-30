import zarr
from typing import TypedDict

from pydantic_zarr.core import ArraySpec, GroupSpec, from_spec, to_spec


def test_serde():
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
                shape=(1000,),
                chunks=(100,),
                dtype="uint8",
                attrs=ArrayAttrs(scale=[1.0]),
            ),
            "s1": ArraySpec(
                shape=(500,),
                chunks=(100,),
                dtype="uint8",
                attrs=ArrayAttrs(scale=[2.0]),
            ),
            "subgroup": GroupSpec(attrs=SubGroupAttrs(a="foo", b=1.0)),
        },
    )

    # materialize a zarr group, based on the spec
    group = from_spec(store, "/group_a", spec)

    # parse the spec from that group
    name, observed = to_spec(group)
    assert name == "group_a"
    assert observed == spec
