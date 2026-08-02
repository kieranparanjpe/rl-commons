from __future__ import annotations

import json
from itertools import product
from typing import Any

from pydantic import TypeAdapter


class ConfigLoader:

    @classmethod
    def load_single(cls, path: str, sections: dict[str, type]) -> dict[str, Any]:
        data = cls._load_data(path)
        return cls._parse_single(data, sections)

    @classmethod
    def load_grid(cls, path: str, sections: dict[str, type]) -> list[dict[str, Any]]:
        data = cls._load_data(path)
        return cls._parse_grid(data, sections)

    @classmethod
    def _parse_single(cls, data: dict, sections: dict[str, type]) -> dict[str, Any]:
        return {
            name: TypeAdapter(cls_).validate_python(data.get(name, {}))
            for name, cls_ in sections.items()
        }

    @classmethod
    def _parse_grid(cls, data: dict, sections: dict[str, type]) -> list[dict[str, Any]]:
        """
        Values that are lists become grid dimensions; scalars are fixed.
        The final grid is the cartesian product across all dimensions from all sections.
        """
        section_names = list(sections.keys())
        grid_dims: list[tuple[str, str, list]] = []  # (section, key, values)
        fixed: dict[str, dict] = {s: {} for s in section_names}

        for section in section_names:
            for key, val in data.get(section, {}).items():
                if isinstance(val, list):
                    grid_dims.append((section, key, val))
                else:
                    fixed[section][key] = val

        all_keys = [(s, k) for s, k, _ in grid_dims]
        all_vals = [v for _, _, v in grid_dims]

        configs = []
        for combo in product(*all_vals):
            section_dicts: dict[str, dict] = {s: dict(fixed[s]) for s in section_names}
            for (section, key), value in zip(all_keys, combo):
                section_dicts[section][key] = value
            configs.append(cls._parse_single(section_dicts, sections))

        return configs

    @classmethod
    def _load_data(cls, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
