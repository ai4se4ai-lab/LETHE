from shop.checkout import checkout


def test_new_customer_small_order():
    assert checkout([("A", 1)], 0) == {"total": 60.0, "status": "auto"}


def test_one_year_member():
    assert checkout([("A", 2)], 1) == {"total": 95.0, "status": "auto"}


def test_two_year_member():
    assert checkout([("B", 1)], 2) == {"total": 95.0, "status": "auto"}


def test_one_year_small_order():
    assert checkout([("A", 1)], 1) == {"total": 57.5, "status": "auto"}


def test_two_year_mixed_cart():
    assert checkout([("A", 1), ("C", 2)], 2) == {"total": 95.48, "status": "auto"}


def test_two_year_big_cart():
    assert checkout([("B", 2)], 2) == {"total": 190.0, "status": "review"}


def test_one_year_two_items():
    assert checkout([("A", 1), ("B", 1)], 1) == {"total": 142.5, "status": "auto"}


def test_four_year_member():
    assert checkout([("B", 1)], 4) == {"total": 90.0, "status": "auto"}


def test_five_year_member():
    assert checkout([("A", 2)], 5) == {"total": 90.0, "status": "auto"}


def test_bulk_order():
    assert checkout([("A", 10)], 0) == {"total": 490.0, "status": "review"}


def test_rounding_case():
    assert checkout([("C", 3)], 0) == {"total": 69.97, "status": "auto"}


def test_three_year_member():
    assert checkout([("A", 2), ("B", 1)], 3) == {"total": 180.0, "status": "review"}
