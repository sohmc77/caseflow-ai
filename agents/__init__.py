"""Framework-agnostic agent layer.

Nothing in this package imports Django. Agents take an immutable CaseSnapshot
and return validated Pydantic contracts, so they can be unit-tested and
evaluated without a database.
"""
