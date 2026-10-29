"""Build the compiled A* and waypoint selection of one_scan_is_enough.planning.

Everything else is configured in pyproject.toml.
"""

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

setup(
    ext_modules=[
        Pybind11Extension(
            "one_scan_is_enough.planning._native",
            ["src/one_scan_is_enough/planning/native.cpp"],
            cxx_std=17,
            extra_compile_args=["-O3", "-ffp-contract=off"],
        )
    ]
)
