from pydantic import BaseModel

from typing import TypeVar, Union, Generic

T = TypeVar("T", bound=Union["A", "B"])


class A(BaseModel):
    shared_prop: str
    a_prop: str


class B(BaseModel):
    shared_prop: str
    b_prop: str


class C(BaseModel, Generic[T]):
    members: dict[str, T]


def test_ab():
    a = A(shared_prop="hi", a_prop="i am a")
    b = B(shared_prop="hi", b_prop="i am b")
    c1 = C(members={"a": a, "b": b})
    c2 = C(**c1.model_dump())
    assert c1 == c2
