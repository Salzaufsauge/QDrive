import collections.abc
import inspect
import numbers
import typing
import uuid

from nicegui import ui

from util.inspection_helper import unwrap_optional


def build_ui_params(params: list, elem_per_row: int, action) -> dict:
    elems = {}

    for i in range(0, len(params), elem_per_row):
        with ui.row().classes("w-full"):
            for param in params[i : i + elem_per_row]:
                elems[param.name] = action(param=param)

    return elems


def add_table(table_name, table_data):
    for row in table_data:
        row["_id"] = str(uuid.uuid4())

    with ui.column().classes("flex-grow"):
        ui.label(table_name).classes("text-lg font-bold")
        grid = ui.aggrid(
            {
                "columnDefs": [
                    {"headerName": "_id", "field": "_id", "hide": True},
                    {"name": "key", "label": "key", "field": "key", "editable": True},
                    {
                        "name": "value",
                        "label": "value",
                        "field": "value",
                        "editable": True,
                    },
                ],
                "rowData": table_data,
                "rowSelection": "multiple",
                "stopEditingWhenCellsLoseFocus": True,
            },
            auto_size_columns=True,
        ).classes("flex-grow")
        with ui.row().classes("flex-grow"):
            add_btn = ui.button("Add Row").classes("flex-grow")
            rm_btn = ui.button("Remove selected Rows").classes("flex-grow")

    def add_row():
        new_id = str(uuid.uuid4())
        grid.options["rowData"].append({"_id": new_id, "key": None, "value": None})

    def handle_cell_value_change(e):
        new_row = e.args["data"]
        grid.options["rowData"][:] = [
            row | new_row if row["_id"] == new_row["_id"] else row
            for row in grid.options["rowData"]
        ]

    async def delete_selected():
        selected_id = [row["_id"] for row in await grid.get_selected_rows()]
        grid.options["rowData"][:] = [
            row for row in grid.options["rowData"] if row["_id"] not in selected_id
        ]

    grid.on("cellValueChanged", handle_cell_value_change)
    add_btn.on_click(add_row)
    rm_btn.on_click(delete_selected)

    return grid


def make_ui_for_param(param, value=None, visible=True):
    ann = param.annotation
    args = typing.get_args(ann)

    val = param.default if value is None else value
    val = None if val is inspect.Parameter.empty else val

    if callable(val):
        val = inspect.getsource(val).strip()

    if args:
        ann = unwrap_optional(ann)

        if any(
            typing.get_origin(a) is collections.abc.Callable
            or a in (collections.abc.Callable, typing.Callable)
            for a in args
        ):
            elem = ui.input(label=param.name, value=str(val) if val is not None else "")

            elem.set_visibility(visible)
            return elem.classes("flex-grow")

    origin = typing.get_origin(ann)

    if ann is str:
        elem = ui.input(label=param.name, value=val)

    elif ann is int or ann is float:
        elem = ui.number(
            label=param.name,
            value=val,
        )

    elif ann is bool:
        elem = ui.checkbox(text=param.name, value=val)

    elif origin is dict:
        rows = [{"key": k, "value": v} for k, v in (val or {}).items()]

        elem = add_table(param.name, rows)

    elif origin is typing.Callable:
        elem = ui.textarea(label=param.name, value=str(val) if val is not None else "")

    elif isinstance(val, bool):
        elem = ui.checkbox(text=param.name, value=val)

    elif isinstance(val, numbers.Number):
        elem = ui.number(
            label=param.name,
            value=val,
        )

    elif param.name.endswith("keys"):
        rows = [{"key": k, "value": v} for k, v in (val or {}).items()]

        elem = add_table(param.name, rows)

    else:
        elem = ui.input(
            label=f"{param.name} (unknown type)",
            value=str(val) if val is not None else "",
        )

    elem.set_visibility(visible)
    return elem.classes("flex-grow")


def unwrap_ui_elem(elem):
    if isinstance(elem, ui.select) and hasattr(elem, "noise_sigma"):
        if not elem.value:
            return None
        spec = {"type": elem.value, "sigma": elem.noise_sigma.value}
        if elem.value == "OrnsteinUhlenbeckActionNoise":
            spec["theta"] = elem.noise_theta.value
        return spec

    if isinstance(
        elem, (ui.input, ui.checkbox, ui.number, ui.textarea, ui.select, ui.input_chips)
    ):
        return elem.value
    if isinstance(elem, ui.label):
        return elem.text
    if isinstance(elem, ui.aggrid):
        return {row["key"]: row["value"] for row in elem.options["rowData"]}

    raise ValueError(f"Not a known ui element: {type(elem).__name__}")
