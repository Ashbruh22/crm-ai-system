"""Agentic decision layer (spec section 8).

``nba.py`` holds the hierarchical decision tree as explicit, individually
testable rules. Every recommendation records which rule fired and why, which is
what makes the ``actions`` table an audit trail rather than a list of guesses.
"""
