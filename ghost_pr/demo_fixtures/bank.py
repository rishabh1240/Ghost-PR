"""Demo fixture: a simple withdraw function with a boundary bug gap."""

# ── This file is bundled as a demo fixture for `ghost haunt --demo`. ──
# It is copied into a temporary workspace during mutation testing.
# The original is NEVER modified.


def withdraw(balance, amount):
    """Withdraw *amount* from *balance*, or raise on insufficient funds."""
    if amount <= balance:
        return balance - amount
    raise ValueError("Insufficient funds")
