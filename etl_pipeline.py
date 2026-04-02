"""ETL Pipeline — Amman Digital Market Customer Analytics

Extracts data from PostgreSQL, transforms it into customer-level summaries,
validates data quality, and loads results to a database table and CSV file.
"""
from sqlalchemy import create_engine
import pandas as pd
import os


def extract(engine):
    """Extract all source tables from PostgreSQL into DataFrames.

    Args:
        engine: SQLAlchemy engine connected to the amman_market database

    Returns:
        dict: {"customers": df, "products": df, "orders": df, "order_items": df}
    """
    tables = ["customers", "products", "orders", "order_items"]
    data = {}

    for table in tables:
        data[table] = pd.read_sql(f"SELECT * FROM {table}", engine)
        print(f"Extracted {table}: {len(data[table])} rows")

    return data


def transform(data_dict):
    """Transform raw data into customer-level analytics summary.

    Steps:
    1. Join orders with order_items and products
    2. Compute line_total (quantity * unit_price)
    3. Filter out cancelled orders (status = 'cancelled')
    4. Filter out suspicious quantities (quantity > 100)
    5. Aggregate to customer level: total_orders, total_revenue,
       avg_order_value, top_category

    Args:
        data_dict: dict of DataFrames from extract()

    Returns:
        DataFrame: customer-level summary with columns:
            customer_id, customer_name, city, total_orders,
            total_revenue, avg_order_value, top_category
    """
    customers = data_dict["customers"].copy()
    products = data_dict["products"].copy()
    orders = data_dict["orders"].copy()
    order_items = data_dict["order_items"].copy()

    # Rename conflicting columns before merge
    customers = customers.rename(columns={"name": "customer_name"})
    products = products.rename(columns={"name": "product_name"})

    # Filter out cancelled orders
    orders = orders[orders["status"].str.lower() != "cancelled"].copy()

    # Filter out suspicious quantities
    order_items = order_items[order_items["quantity"] <= 100].copy()

    # Join orders with order_items
    merged = orders.merge(order_items, on="order_id", how="inner")

    # Join with products
    merged = merged.merge(products, on="product_id", how="inner")

    # Join with customers
    merged = merged.merge(customers, on="customer_id", how="inner")

    # Compute line_total
    merged["line_total"] = merged["quantity"] * merged["unit_price"]

    # Aggregate customer summary
    summary = (
        merged.groupby(["customer_id", "customer_name", "city"], dropna=False)
        .agg(
            total_orders=("order_id", pd.Series.nunique),
            total_revenue=("line_total", "sum"),
        )
        .reset_index()
    )

    summary["avg_order_value"] = (
        summary["total_revenue"] / summary["total_orders"]
    ).round(2)

    # Calculate top category by revenue for each customer
    category_revenue = (
        merged.groupby(["customer_id", "category"], dropna=False)["line_total"]
        .sum()
        .reset_index()
    )

    category_revenue = category_revenue.sort_values(
        by=["customer_id", "line_total", "category"],
        ascending=[True, False, True]
    )

    top_category = (
        category_revenue.groupby("customer_id", as_index=False)
        .first()[["customer_id", "category"]]
        .rename(columns={"category": "top_category"})
    )

    summary = summary.merge(top_category, on="customer_id", how="left")

    summary = summary[
        [
            "customer_id",
            "customer_name",
            "city",
            "total_orders",
            "total_revenue",
            "avg_order_value",
            "top_category",
        ]
    ]

    summary["total_revenue"] = summary["total_revenue"].round(2)

    return summary


def validate(df):
    """Run data quality checks on the transformed DataFrame.

    Checks:
    - No nulls in customer_id or customer_name
    - total_revenue > 0 for all customers
    - No duplicate customer_ids
    - total_orders > 0 for all customers

    Args:
        df: transformed customer summary DataFrame

    Returns:
        dict: {check_name: bool} for each check

    Raises:
        ValueError: if any critical check fails
    """
    results = {
        "no_null_customer_id": df["customer_id"].notna().all(),
        "no_null_customer_name": df["customer_name"].notna().all(),
        "total_revenue_positive": (df["total_revenue"] > 0).all(),
        "no_duplicate_customer_id": ~df["customer_id"].duplicated().any(),
        "total_orders_positive": (df["total_orders"] > 0).all(),
    }

    for check_name, passed in results.items():
        print(f"{check_name}: {'PASS' if passed else 'FAIL'}")

    if not all(results.values()):
        failed_checks = [k for k, v in results.items() if not v]
        raise ValueError(f"Validation failed for: {', '.join(failed_checks)}")

    return results


def load(df, engine, csv_path):
    """Load customer summary to PostgreSQL table and CSV file.

    Args:
        df: validated customer summary DataFrame
        engine: SQLAlchemy engine
        csv_path: path for CSV output
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    df.to_sql("customer_analytics", engine, if_exists="replace", index=False)
    df.to_csv(csv_path, index=False)

    print(f"Loaded {len(df)} rows to database table 'customer_analytics'")
    print(f"Saved CSV to {csv_path}")


def main():
    """Orchestrate the ETL pipeline: extract -> transform -> validate -> load."""
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5432/amman_market"
    )

    print("Creating database engine...")
    engine = create_engine(database_url)

    print("Starting extract stage...")
    data = extract(engine)

    print("Starting transform stage...")
    customer_summary = transform(data)
    print(f"Transformed rows: {len(customer_summary)}")

    print("Starting validation stage...")
    validate(customer_summary)

    print("Starting load stage...")
    load(customer_summary, engine, "output/customer_analytics.csv")

    print("ETL pipeline completed successfully.")


if __name__ == "__main__":
    main()