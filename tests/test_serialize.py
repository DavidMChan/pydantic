# Copyright (c) 2022 David Chan
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import datetime
import json
import re
import sys
from dataclasses import dataclass as vanilla_dataclass
from decimal import Decimal
from enum import Enum
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from pathlib import Path
from typing import List, Optional
from uuid import UUID

import pytest

from pydantic import BaseModel, NameEmail, create_model
from pydantic.color import Color
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic.json import pydantic_encoder, timedelta_isoformat
from pydantic.types import ConstrainedDecimal, DirectoryPath, FilePath, SecretBytes, SecretStr

def test_model_serialize():
    class ModelA(BaseModel):
        x: int
        y: str

    class Model(BaseModel):
        a: float
        b: bytes
        c: Decimal
        d: ModelA

    m = Model(a=10.2, b='foobar', c=10.2, d={'x': 123, 'y': '123'})
    assert m.dict(serialize=True) == {"a": 10.2, "b": "foobar", "c": 10.2, "d": {"x": 123, "y": "123"}}
    assert json.dumps(m.dict(serialize=True)) == '{"a": 10.2, "b": "foobar", "c": 10.2, "d": {"x": 123, "y": "123"}}'
    assert m.dict(serialize=True, exclude={'b'}) == {"a": 10.2, "c": 10.2, "d": {"x": 123, "y": "123"}}

def test_subclass_serialize():
    class SubDate(datetime.datetime):
        pass

    class Model(BaseModel):
        a: datetime.datetime
        b: SubDate

    m = Model(a=datetime.datetime(2032, 1, 1, 1, 1), b=SubDate(2020, 2, 29, 12, 30))
    assert m.dict(serialize=True) == {"a": "2032-01-01T01:01:00", "b": "2020-02-29T12:30:00"}

def test_subclass_custom_serialize():
    class SubDate(datetime.datetime):
        pass

    class SubDelta(datetime.timedelta):
        pass

    class Model(BaseModel):
        a: SubDate
        b: SubDelta

        class Config:
            json_encoders = {
                datetime.datetime: lambda v: v.strftime('%a, %d %b %C %H:%M:%S'),
                datetime.timedelta: timedelta_isoformat,
            }

    m = Model(a=SubDate(2032, 1, 1, 1, 1), b=SubDelta(hours=100))
    assert m.dict(serialize=True) == {"a": "Thu, 01 Jan 20 01:01:00", "b": "P4DT4H0M0.000000S"}

def test_custom_model_serialize():
    class Model(BaseModel):
        x: datetime.timedelta
        y: Decimal
        z: datetime.date

        class Config:
            json_encoders = {datetime.timedelta: lambda v: f'{v.total_seconds():0.3f}s', Decimal: lambda v: 'a decimal'}

    assert Model(x=123, y=5, z='2032-06-01').dict(serialize=True) == {"x": "123.000s", "y": "a decimal", "z": "2032-06-01"}

### --

def test_custom_iso_timedelta_serialize():
    class Model(BaseModel):
        x: datetime.timedelta

        class Config:
            json_encoders = {datetime.timedelta: timedelta_isoformat}

    m = Model(x=123)
    assert m.dict(serialize=True) == {"x": "P0DT0H2M3.000000S"}


def test_con_decimal_serialize() -> None:
    """
    Makes sure a decimal with decimal_places = 0, as well as one with places
    can handle a encode/decode roundtrip.
    """

    class Id(ConstrainedDecimal):
        max_digits = 22
        decimal_places = 0
        ge = 0

    class Obj(BaseModel):
        id: Id
        price: Decimal = Decimal('0.01')

    assert Obj(id=1).dict(serialize=True) == {"id": 1, "price": 0.01}
    assert Obj.parse_obj(Obj(id=1).dict(serialize=True)) == Obj(id=1)


def test_dict_serialize_simple_inheritance():
    class Parent(BaseModel):
        dt: datetime.datetime = datetime.datetime.now()
        timedt: datetime.timedelta = datetime.timedelta(hours=100)

        class Config:
            json_encoders = {datetime.datetime: lambda _: 'parent_encoder'}

    class Child(Parent):
        class Config:
            json_encoders = {datetime.timedelta: lambda _: 'child_encoder'}

    assert Child().dict(serialize=True) == {"dt": "parent_encoder", "timedt": "child_encoder"}


def test_dict_serialize_inheritance_override():
    class Parent(BaseModel):
        dt: datetime.datetime = datetime.datetime.now()

        class Config:
            json_encoders = {datetime.datetime: lambda _: 'parent_encoder'}

    class Child(Parent):
        class Config:
            json_encoders = {datetime.datetime: lambda _: 'child_encoder'}

    assert Child().dict(serialize=True) == {"dt": "child_encoder"}


def test_custom_serialize_arg():
    class Unserializeable:
        pass

    class Model(BaseModel):
        x: Unserializeable

        class Config:
            arbitrary_types_allowed = True

    m = Model(x=Unserializeable())
    with pytest.raises(TypeError):
        m.dict(serialize=True)
    assert m.dict(serialize=lambda v: '__default__') == {"x": "__default__"}


# TODO: Fix this test
def test_serialize_custom_root():
    class Model(BaseModel):
        __root__: List[str]
    assert Model(__root__=['a', 'b']).dict(serialize=True) == ["a", "b"]


def test_custom_decode_encode_serialize():
    load_calls, dump_calls = 0, 0

    def custom_loads(s):
        nonlocal load_calls
        load_calls += 1
        return json.loads(s.strip('$'))

    def custom_dumps(s, default=None, **kwargs):
        nonlocal dump_calls
        dump_calls += 1
        return json.dumps(s, default=default, indent=2)

    class Model(BaseModel):
        a: int
        b: str

        class Config:
            json_loads = custom_loads
            json_dumps = custom_dumps

    m = Model.parse_raw('${"a": 1, "b": "foo"}$$')
    assert m.dict(serialize=True) == {'a': 1, 'b': 'foo'}


def test_json_nested_serialize_models():
    class Phone(BaseModel):
        manufacturer: str
        number: int

    class User(BaseModel):
        name: str
        SSN: int
        birthday: datetime.datetime
        phone: Phone
        friend: Optional['User'] = None  # noqa: F821  # https://github.com/PyCQA/pyflakes/issues/567

        class Config:
            json_encoders = {
                datetime.datetime: lambda v: v.timestamp(),
                Phone: lambda v: v.number if v else None,
                'User': lambda v: v.SSN,
            }

    User.update_forward_refs()

    iphone = Phone(manufacturer='Apple', number=18002752273)
    galaxy = Phone(manufacturer='Samsung', number=18007267864)

    timon = User(
        name='Timon', SSN=123, birthday=datetime.datetime(1993, 6, 1, tzinfo=datetime.timezone.utc), phone=iphone
    )
    pumbaa = User(
        name='Pumbaa', SSN=234, birthday=datetime.datetime(1993, 5, 15, tzinfo=datetime.timezone.utc), phone=galaxy
    )

    timon.friend = pumbaa

    # assert iphone.dict(serialize=True) == {"manufacturer": "Apple", "number": 18002752273}
    assert (
        pumbaa.dict(serialize=True)
        == {"name": "Pumbaa", "SSN": 234, "birthday": 737424000.0, "phone": 18007267864, "friend": None}
    )
    # assert (
    #     timon.dict(serialize=True)
    #     == {"name": "Timon", "SSN": 123, "birthday": 738892800.0, "phone": 18002752273, "friend": 234}
    # )


def test_recursive():
    class Model(BaseModel):
        value: Optional[str]
        nested: Optional[BaseModel]

    assert Model(value=None, nested=Model(value=None)).dict(exclude_none=True, serialize=True) == {"nested": {}}
