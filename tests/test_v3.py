from pydantic_zarr.v3 import ArraySpec
import json


def test_serialize_deserialize():
    json_str = """{
    "zarr_format": 3,
    "node_type": "array",
    "shape": [10000, 1000],
    "dimension_names": ["rows", "columns"],
    "data_type": "float64",
    "chunk_grid": {
        "name": "regular",
        "configuration": {
            "chunk_shape": [1000, 100]
        }
    },
    "chunk_key_encoding": {
        "name": "default",
        "configuration": {
            "separator": "/"
        }
    },
    "codecs": [{
        "name": "gzip",
        "configuration": {
            "level": 1
        }
    }],
    "fill_value": "NaN",
    "attributes": {
        "foo": 42,
        "bar": "apples",
        "baz": [1, 2, 3, 4]
    }
}
    """
    model_dict = json.loads(json_str)
    ArraySpec(**model_dict)
