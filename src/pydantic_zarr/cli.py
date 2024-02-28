from typing import Any, Literal, Tuple, Union, cast
import click
from pydantic_zarr.v2 import ArraySpec, GroupSpec, from_zarr
from pydantic import ValidationError
import zarr
from rich.console import Console
from rich.traceback import install
from rich import print_json
from importlib import import_module

install()


@click.group
def cli():
    pass


def validate(
    cls: GroupSpec | ArraySpec, node: zarr.Array | zarr.Group, console: Console
):
    """
    Attempt to validate a Zarr array or group against a `GroupSpec` or `ArraySpec.
    The result of the validation (success or failure) is printed to the terminal.
    """
    class_name = f"{cls.__module__}.{cls.__name__}"
    protocol = "file"
    if hasattr(node.store, "fs"):
        if isinstance(node.store.fs.protocol, tuple):
            protocol = node.store.fs.protocol[0]
    node_path = protocol + "://" + node.store.path + "/" + node.path
    try:
        cls.from_zarr(node)
        console.print(
            f"[bold]{class_name}[/bold]: [blue]{node_path}[/blue] validated "
            "successfully."
        )
    except ValidationError as e:
        console.print(
            f"[bold]{class_name}[/bold]: [blue]{node_path}[/blue] failed "
            "validation due to the following validation errors:"
        )
        console.print(str(e))


@cli.command
@click.argument("url", type=click.STRING)
@click.argument("group_type", type=click.STRING)
def check(url: str, group_type: str):
    """
    Check whether a Zarr / N5 hierarchy, specified by a url, is consistent with a model,
    specified by a class name.

    Parameters\n
    ----------

    url: A url referring to a Zarr / N5 group or dataset.

    group_type: A class name for one of the models defined in `cellmap-schemas`.
    """
    store_stem, prefix, component_path = parse_url(url)
    store = guess_store(store_stem=store_stem, prefix=prefix)
    node = zarr.open(store, path=component_path, mode="r")

    console = Console()

    *mod_path_parts, classname = group_type.split(".")
    mod = import_module(".".join([__package__] + mod_path_parts))
    cls: Any = getattr(mod, classname)
    if not issubclass(cls, (ArraySpec, GroupSpec)):
        raise ValueError(
            f"Object of type {type(cls)} is not a `GroupSpec` or `ArraySpec` and thus it does not support validation with this command."
        )
    cls = cast(Union[ArraySpec, GroupSpec], cls)
    validate(cls, node, console)


@cli.command
@click.option("--url", type=click.STRING)
@click.option("--store", type=click.STRING)
@click.option("--path", type=click.STRING)
def inspect(url: str):
    store_stem, prefix, component_path = parse_url(url)
    store = guess_store(store_stem=store_stem, prefix=prefix)
    node = zarr.open(store, path=component_path, mode="r")
    print_json(from_zarr(node).model_dump_json(indent=2))


def guess_store(store_stem: str, prefix: Literal[".zarr"]) -> zarr.storage.BaseStore:
    store = zarr.storage.FSStore(store_stem + prefix)
    return store


def parse_url(url: str) -> Tuple[str, Literal[".zarr"], str]:
    """
    Parses a string into a pre-suffix component, a suffix, and a post-suffix component, for the
    suffix ".zarr"

    Parameters
    ----------

    url: str
        An FSSpec-compatible URL, e.g. s3://bucket/path.zarr/foo or http://domain.com/path.n5/bar

    Returns
    -------
    Tuple[str, Literal['.zarr'], str]
    """

    # try zarr
    if ".zarr" not in url:
        raise ValueError(
            f"Invalid url parameter. Got {url}, but expected a string containing a at "
            "least one instance of '.zarr'. Are you sure this url refers to "
            "Zarr"
        )

    return url.partition(".zarr")
