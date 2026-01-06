# Adapted from https://github.com/peterroelants/annotated-docs

import inspect
from collections.abc import Callable
from typing import Any, Final, TypedDict, TypeVar #Required, 
from typing_extensions import Required

import pydantic
import pydantic.json_schema

RETURNS_KEY: Final[str] = "returns"

T = TypeVar("T")


class FunctionJSONSchema(TypedDict, total=False):
    name: Required[str]
    description: str
    parameters: dict[str, Any]


def as_json_schema(func: Callable) -> FunctionJSONSchema:
    """
    Return a JSON schema for the given function.
    """
    parameters_schema = get_parameters_schema(func)
    description = ""
    if func.__doc__:
        description = inspect.cleandoc(func.__doc__).strip()
    schema_dct: FunctionJSONSchema = {
        "name": func.__name__,
        "description": description,
        "parameters": parameters_schema,
    }
    return schema_dct


def doc(description) -> Any:
    """Annotate a variable with a description."""
    return pydantic.Field(description=description)


def get_parameters_schema(func: Callable) -> dict[str, Any]:
    """Return a JSON schema for the parameters of the given function."""
    parameter_model = get_parameter_model(func)
    return parameter_model.model_json_schema(
        schema_generator=GenerateJsonSchemaNoTitle,
        mode="validation",
    )


def get_parameter_model(func: Callable) -> pydantic.BaseModel:
    """
    Return a Pydantic model for the parameters of the given function.
    """
    field_definitions: dict[str, tuple[Any, Any]] = {}
    for name, obj in inspect.signature(func).parameters.items():
        if obj.annotation == inspect.Parameter.empty:
            raise ValueError(
                f"`{func.__name__}` parameter `{name!s}` has no annotation, please provide an notation to be able to generate the function specification."
            )
        if obj.default == inspect.Parameter.empty:
            field_definitions[name] = (obj.annotation, pydantic.Field(...))
        else:
            field_definitions[name] = (obj.annotation, obj.default)
    _model_name = ""  # Empty model name
    return pydantic.create_model(_model_name, **field_definitions)  # type: ignore


def get_returns_schema(func: Callable) -> dict[str, Any]:
    returns_model = get_returns_model(func)
    return_schema = returns_model.model_json_schema(
        schema_generator=GenerateJsonSchemaNoTitle,
        mode="validation",
    )
    properties = return_schema.pop("properties")
    return_schema |= properties[RETURNS_KEY]
    if "required" in return_schema:
        del return_schema["required"]
    if "type" in return_schema and return_schema["type"] == "object":
        del return_schema["type"]
    return return_schema


def get_returns_model(func: Callable) -> pydantic.BaseModel:
    """
    Return a Pydantic model for the returns of the given function.
    """
    return_annotation = inspect.signature(func).return_annotation
    if return_annotation == inspect.Signature.empty:
        raise ValueError(
            f"`{func.__name__}` has no return annotation, please provide an annotation to be able to generate the function specification."
        )
    field_definitions: dict[str, tuple[Any, Any]] = {
        RETURNS_KEY: (return_annotation, pydantic.Field(...))
    }
    _model_name = ""  # Empty model name
    return pydantic.create_model(_model_name, **field_definitions)  # type: ignore


class GenerateJsonSchemaNoTitle(pydantic.json_schema.GenerateJsonSchema):
    def generate(
        self, schema, mode="validation"
    ) -> pydantic.json_schema.JsonSchemaValue:
        json_schema = super().generate(schema, mode=mode)
        if "title" in json_schema:
            del json_schema["title"]
        return json_schema

    def get_schema_from_definitions(
        self, json_ref
    ) -> pydantic.json_schema.JsonSchemaValue | None:
        json_schema = super().get_schema_from_definitions(json_ref)
        if json_schema and "title" in json_schema:
            del json_schema["title"]
        return json_schema

    def field_title_should_be_set(self, schema) -> bool:
        return False