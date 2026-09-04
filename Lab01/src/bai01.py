"""Bai 1 - Kiem tra moi truong Python.

In phien ban Python, Gymnasium, NumPy (va Matplotlib) bang cach doc truc tiep
tu thu vien, khong duoc go tay so phien ban.
"""

import sys

import gymnasium as gym
import matplotlib
import numpy as np


def main() -> None:
    # sys.version_info cho phien ban Python dang chay (khong hard-code)
    py_version = ".".join(str(part) for part in sys.version_info[:3])

    print(f"Python version    : {py_version}")
    print(f"Gymnasium version : {gym.__version__}")
    print(f"NumPy version     : {np.__version__}")
    print(f"Matplotlib version: {matplotlib.__version__}")

    # In them chuoi day du cua Python de tien kiem tra
    print()
    print("sys.version:", sys.version.replace("\n", " "))


if __name__ == "__main__":
    main()
