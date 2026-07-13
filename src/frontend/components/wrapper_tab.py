from __future__ import annotations

import inspect
from typing import Callable

from nicegui import ui

from util.inspection_helper import load_env_wrappers, make_ui_for_param
from util.utils import build_ui_params

_dragged: Draggable | None = None


def make_wrapper_ui(param: inspect.Parameter):
    if param.name in ["env", "venv"]:
        elem = ui.label("skip")
        elem.set_visibility(False)
        return elem

    return make_ui_for_param(param)


class WrapperTab:
    def __init__(self):
        self.wrappers = load_env_wrappers()
        self.wrapper_items = []
        self.wrapper_container = None
        self.search = ""
        self.only_active = None
        self.parent_filter = None
        self.module_filter = None

    def build(self):
        self.wrapper_items = []

        with ui.row().classes("w-full"):
            ui.checkbox("Only active wrappers", on_change=lambda e: self.set_filter_active(e.value)).classes(
                "flex-grow")
            ui.input("Search wrappers", on_change=lambda e: self.apply_filters(e.value)).classes("flex-grow")
            self.parent_filter = ui.select(options=[], label="Filter wrapper type", multiple=True, clearable=True,
                                           on_change=lambda e: self.apply_filters()).classes("flex-grow")
            self.module_filter = ui.select(options=[], label="Filter module", multiple=True, clearable=True,
                                           on_change=lambda e: self.apply_filters()).classes("flex-grow")

        with Column().classes('w-full') as self.wrapper_container:
            for wrapper_name, wrapper_cls in self.wrappers.items():
                self.add_wrapper(wrapper_name, wrapper_cls)
        self.parent_filter.set_options(self.get_available_parents())
        self.module_filter.set_options(self.available_modules())

    @property
    def wrapper_params(self):
        result = []

        for item in self.wrapper_items:
            result.append(item["checkbox"])

            if item["active"]:
                result.extend(item["params"])

        return result

    def add_wrapper(self, wrapper_name, wrapper_cls):
        item = {
            "name": wrapper_name,
            "cls": wrapper_cls,
            "module": wrapper_cls.__module__,
            "parents": [
                cls.__name__
                for cls in wrapper_cls.__mro__[1:]
                if cls is not object
            ],
            "active": False,
            "params": [],
            "element": None,
        }

        with Draggable(width_class="w-full", on_drag_finished=self.drag_finished) as card:
            item["element"] = card
            with ui.expansion(text=wrapper_name).classes("w-full"):
                cb = ui.checkbox(
                    "Enable " + wrapper_name,
                    value=False,
                    on_change=lambda e, i=item: self.set_active(i, e.value)
                )
                item["checkbox"] = cb
                params = list(
                    inspect.signature(wrapper_cls).parameters.values()
                )
                item["params"] = build_ui_params(
                    params,
                    4,
                    make_wrapper_ui
                )

        self.wrapper_items.append(item)

    def set_active(self, item, active):
        item["active"] = active

        if active:
            item["element"].enable_drag()
        else:
            item["element"].disable_drag()

        self.sort_wrappers()
        self.update_ui_order()

    def sort_wrappers(self):
        active = [
            x for x in self.wrapper_items
            if x["active"]
        ]

        inactive = [
            x for x in self.wrapper_items
            if not x["active"]
        ]

        self.wrapper_items = active + inactive

    def update_ui_order(self):
        with self.wrapper_container:
            for item in self.wrapper_items:
                item["element"].move()

    def drag_finished(self, e: tuple[int, int]):
        item = self.wrapper_items.pop(e[0])
        self.wrapper_items.insert(e[1], item)

        ui.notify(
            f"{item['name']} moved"
        )

    def available_modules(self):
        return sorted({
            item["module"]
            for item in self.wrapper_items
        })

    def get_available_parents(self):
        parents = set()

        for item in self.wrapper_items:
            parents.update(item["parents"])

        return list(parents)

    def set_filter_active(self, active):
        self.only_active = active
        self.apply_filters()

    def apply_filters(self, text=None):
        if text is not None:
            self.search = text.lower()

        parents = (
            self.parent_filter.value
            if hasattr(self, "parent_filter")
            else None
        )

        modules = (
            self.module_filter.value
            if hasattr(self, "module_filter")
            else None
        )

        for item in self.wrapper_items:
            visible = True

            if self.search:
                visible &= (
                        self.search in item["name"].lower()
                )

            if self.only_active:
                visible &= item["active"]

            if parents:
                visible &= any(parent in item["parents"] for parent in parents)

            if modules:
                visible &= any(module == item["module"] for module in modules)

            item["element"].set_visibility(visible)


# Taken from https://github.com/zigai/nicegui-extensions and slightly modified
class Draggable(ui.card):
    dragged_background = "bg-blue-950"

    def __init__(self, enable_dragging: bool = False, on_drag_finished: Callable | None = None,
                 width_class="w-fit-content") -> None:
        super().__init__()
        self.drag_enabled = enable_dragging
        self.width_class = width_class
        self.on_drag_finished = on_drag_finished
        self.old_index = None
        self.new_index = None

        with self.props("draggable").classes(f"cursor-pointer").classes(self.width_class).style(
                "box-shadow: none;"
        ):
            self.build()

        self.on("dragstart", self.on_dragstart)
        self.on("dragenter", self.on_dragenter)
        self.on("dragleave", self.on_dragleave)
        self.on("dragover.prevent", self.on_dragover_prevent)
        self.on("drop", self.on_drop)

        if not self.drag_enabled:
            self.disable_drag()

    def disable_drag(self) -> None:
        global _dragged

        if _dragged is self:
            _dragged = None

        self.drag_enabled = False
        self.classes(remove="cursor-pointer")
        self.props(remove="draggable")

    def enable_drag(self) -> None:
        self.drag_enabled = True
        self.classes(add="cursor-pointer")
        self.props(add="draggable")

    def get_parent(self) -> Column | None:
        return self.parent_slot.parent  # type:ignore

    def delete_from_parent(self) -> None:
        parent = self.get_parent()
        if parent is not None:
            parent.remove(self)

    def build(self) -> None:
        """Override this method to build the draggable card"""

    def on_dragstart(self) -> None:
        if not self.drag_enabled:
            return

        self.classes(add=self.dragged_background)

        global _dragged
        _dragged = self

    def on_dragenter(self) -> None:
        if not self.drag_enabled:
            return

    def on_dragleave(self) -> None:
        if not self.drag_enabled:
            return

    def on_drop(self) -> None:
        self.classes(remove=self.dragged_background)
        if not self.drag_enabled:
            return
        global _dragged
        if not _dragged:
            return

        if not _dragged.drag_enabled:
            return

        parent = self.get_parent()
        if parent is None:
            return

        children = parent.get_draggable_children()
        try:
            self.old_index = children.index(_dragged)
            self.new_index = children.index(self)
        except ValueError:
            return

        self.on_drag_finished((self.old_index, self.new_index))
        _dragged.move(target_index=self.new_index)
        self.on_dragleave()
        _dragged.classes(remove=self.dragged_background)

    def on_dragover_prevent(self):
        """Prevent default dragover event to allow drop event"""
        return


class Column(ui.column):
    def get_draggable_children(self) -> list[Draggable]:
        children = []
        for i in self.default_slot.children:
            if issubclass(i.__class__, Draggable):
                if i.drag_enabled:  # type:ignore
                    children.append(i)
        return children
