from shop.catalog import price_of
from shop.pricing import bulk_rate, loyalty_rate

SHIPPING_FEE = 10.0
FREE_SHIPPING_THRESHOLD = 100.0
REVIEW_THRESHOLD = 200.0
TAX_RATE = 0.0


def round_money(x):
    return round(x + 1e-9, 2)


def subtotal(cart):
    total = 0.0
    for sku, qty in cart:
        total += price_of(sku) * qty
    return total


def shipping_fee(amount):
    if amount >= FREE_SHIPPING_THRESHOLD:
        return 0.0
    return SHIPPING_FEE


def risk_review(amount):
    if amount >= REVIEW_THRESHOLD:
        return "review"
    return "auto"


def apply_coupon(amount, coupon=None):
    if coupon is None:
        return amount
    return amount - coupon


def tax_for(amount):
    return amount * TAX_RATE


def packaging_fee(cart):
    return 0.0


def compute_total(cart, years, coupon=None):
    amount = subtotal(cart)
    qty = sum(q for _, q in cart)
    rate = loyalty_rate(years) + bulk_rate(qty)
    discounted = apply_coupon(amount * (1 - rate), coupon)
    total = discounted + tax_for(discounted) + packaging_fee(cart) + shipping_fee(amount)
    return round_money(total)


def checkout(cart, years):
    amount = subtotal(cart)
    return {"total": compute_total(cart, years), "status": risk_review(amount)}
