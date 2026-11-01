"""Pluggable components. Every module here is imported automatically by qam.registry.

Add a new implementation by writing a function or class in the matching file and
decorating it with @register(kind, name). See docs/PLATFORM.md for each contract.
"""
