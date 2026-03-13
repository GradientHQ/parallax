"""
Parallax Doctor - Environment diagnostic module.

This module provides diagnostic checks for validating the Parallax environment
before running the application. It checks Python version, hardware availability,
dependencies, and WSL path issues.

Usage:
    parallax doctor      # Run all diagnostic checks
    parallax doctor -v   # Run with verbose output
"""

from parallax.doctor.checks import run_all_checks

__all__ = ["run_all_checks"]
