def loyalty_rate(years):
    """Discount rate for a customer, based on years of membership."""
    if years > 3:
        return 0.10
    if years >= 1:
        return 0.05
    return 0.0


def bulk_rate(total_qty):
    """Extra discount for large quantities."""
    if total_qty >= 10:
        return 0.02
    return 0.0
