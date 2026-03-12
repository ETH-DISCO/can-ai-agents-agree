#!/usr/bin/env python3
"""Launcher - delegates to byzantine_consensus_game/main.py"""
import subprocess
import sys
import os

script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "byzantine_consensus_game")
sys.exit(subprocess.call(
    [sys.executable, os.path.join(script_dir, "main.py")] + sys.argv[1:],
    cwd=script_dir,
))
