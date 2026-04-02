import pandas as pd
import pytest
from etl_pipeline import transform, validate


def build_test_data():
    customers = pd.DataFrame({
        "customer_id": [1, 2],
        "name": ["Ali", "Sara"],
        "city": ["Amman", "Zarqa"],
        "registration_date": ["2024-01-01", "2024-01-02"]
    })

    products = pd.DataFrame({
        "product_id": [101, 102],
        "name": ["Laptop", "Phone"],
        "category": ["Electronics", "Mobile"],
        "unit_price": [500.0, 300.0]
    })

    orders = pd.DataFrame({
        "order_id": [1001, 1002],
        "customer_id": [1, 2],
        "order_date": ["2024-02-01", "2024-02-02"],
        "status": ["completed", "cancelled"]
    })

    order_items = pd.DataFrame({
        "item_id": [1, 2],
        "order_id": [1001, 1002],
        "product_id": [101, 102],
        "quantity": [2, 1]
    })

    return {
        "customers": customers,
        "products": products,
        "orders": orders,
        "order_items": order_items
    }


def test_transform_filters_cancelled():
    data = build_test_data()
    result = transform(data)

    assert len(result) == 1
    assert result.iloc[0]["customer_id"] == 1
    assert result.iloc[0]["customer_name"] == "Ali"


def test_transform_filters_suspicious_quantity():
    data = build_test_data()
    data["orders"] = pd.DataFrame({
        "order_id": [1001],
        "customer_id": [1],
        "order_date": ["2024-02-01"],
        "status": ["completed"]
    })
    data["order_items"] = pd.DataFrame({
        "item_id": [1],
        "order_id": [1001],
        "product_id": [101],
        "quantity": [150]
    })

    result = transform(data)
    assert result.empty


def test_validate_catches_nulls():
    bad_df = pd.DataFrame({
        "customer_id": [1, None],
        "customer_name": ["Ali", "Sara"],
        "city": ["Amman", "Zarqa"],
        "total_orders": [1, 1],
        "total_revenue": [100.0, 200.0],
        "avg_order_value": [100.0, 200.0],
        "top_category": ["Electronics", "Mobile"]
    })

    with pytest.raises(ValueError):
        validate(bad_df)