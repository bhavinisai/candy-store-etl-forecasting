from pyspark.sql import SparkSession, DataFrame
from data_processor import DataProcessor, process_day_transactions, save_to_csv
from dotenv import load_dotenv
from datetime import datetime, timedelta
import os
from pyspark.sql.functions import col
from typing import Dict, Tuple
import traceback
import numpy as np
import glob
import shutil
from pyspark.sql.functions import sum as spark_sum, round as spark_round, when


def create_spark_session(app_name: str = "CandyStoreAnalytics") -> SparkSession:
    """
    Create and configure Spark session with MongoDB and MySQL connectors
    """
    return (
        SparkSession.builder.appName("CandyStore")
        .config(
            "spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1"
        )
        .config("spark.jars", os.getenv("MYSQL_CONNECTOR_PATH"))
        .config("spark.mongodb.input.uri", os.getenv("MONGODB_URI"))
        .getOrCreate()
    )


def get_date_range(start_date: str, end_date: str) -> list[str]:
    """Generate a list of dates between start and end date"""
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    date_list = []

    current = start
    while current <= end:
        date_list.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    return date_list


def print_header():
    print("*" * 80)
    print("                        CANDY STORE DATA PROCESSING SYSTEM")
    print("                               Analysis Pipeline")
    print("*" * 80)


def print_processing_period(date_range: list):
    print("\n" + "=" * 80)
    print("PROCESSING PERIOD")
    print("-" * 80)
    print(f"Start Date: {date_range[0]}")
    print(f"End Date:   {date_range[-1]}")
    print("=" * 80)


def setup_configuration() -> Tuple[Dict, list]:
    """Setup application configuration"""
    load_dotenv()
    config = load_config()
    date_range = get_date_range(
        os.getenv("MONGO_START_DATE"), os.getenv("MONGO_END_DATE")
    )
    return config, date_range


def load_config() -> Dict:
    """Load configuration from environment variables"""
    return {
        "mongodb_uri": os.getenv("MONGODB_URI"),
        "mongodb_db": os.getenv("MONGO_DB"),
        "mongodb_collection_prefix": os.getenv("MONGO_COLLECTION_PREFIX"),
        "mysql_url": os.getenv("MYSQL_URL"),
        "mysql_user": os.getenv("MYSQL_USER"),
        "mysql_password": os.getenv("MYSQL_PASSWORD"),
        "mysql_db": os.getenv("MYSQL_DB"),
        "customers_table": os.getenv("CUSTOMERS_TABLE"),
        "products_table": os.getenv("PRODUCTS_TABLE"),
        "output_path": os.getenv("OUTPUT_PATH"),
        "mongo_start_date": os.getenv("MONGO_START_DATE"),
        "mongo_end_date": os.getenv("MONGO_END_DATE"),
        "reload_inventory_daily": os.getenv("RELOAD_INVENTORY_DAILY", "false").lower()
        == "true",
    }


def initialize_data_processor(spark: SparkSession, config: Dict) -> DataProcessor:
    """Initialize and configure the DataProcessor"""
    print("\nINITIALIZING DATA SOURCES")
    print("-" * 80)

    data_processor = DataProcessor(spark)
    data_processor.config = config
    return data_processor


def main():
    print_header()

    # Setup
    config, date_range = setup_configuration()
    print_processing_period(date_range)

    # Initialize processor
    spark = create_spark_session()
    data_processor = DataProcessor(spark)

    try:
        # Configure and load data
        data_processor.configure(config)

        output_dir = config["output_path"]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        company_df = data_processor.load_mysql_data(
            config["mysql_url"],
            config["customers_table"],
            config["mysql_user"],
            config["mysql_password"],
        )

        product_df = data_processor.load_mysql_data(
            config["mysql_url"],
            config["products_table"],
            config["mysql_user"],
            config["mysql_password"],
        )
        print(
            f"Customers Table -- Rows: {company_df.count()}, Columns: {len(company_df.columns)}"
        )
        company_df.show(n=5, truncate=False)
        print(
            f"Products Table -- Rows: {product_df.count()}, Columns: {len(product_df.columns)}"
        )
        product_df.show(n=5, truncate=False)

        all_orders = None
        all_items = None

        for date_str in date_range:
            orders_df, items_df = process_day_transactions(spark, config, date_str)

            if orders_df.count() > 0:
                # Append to main DataFrames
                all_orders = (
                    orders_df if all_orders is None else all_orders.union(orders_df)
                )
                all_items = items_df if all_items is None else all_items.union(items_df)

                # Save day-wise files only if data exists
                save_to_csv(
                    orders_df.orderBy("order_id"),
                    config["output_path"],
                    f"batch_orders_{date_str}.csv",
                )
                save_to_csv(
                    items_df.orderBy("order_id", "product_id"),
                    config["output_path"],
                    f"batch_order_line_items_{date_str}.csv",
                )

        save_to_csv(all_orders.orderBy("order_id"), config["output_path"], "orders.csv")
        save_to_csv(
            all_items.orderBy("order_id", "product_id"),
            config["output_path"],
            "order_line_items.csv",
        )

        print("\n Processing daily summary...")
        daily_summary_df = data_processor.calculate_daily_summary(
            all_orders, all_items, product_df
        )
        data_processor.save_to_csv(
            daily_summary_df, config["output_path"], "daily_summary.csv"
        )
        print("\n✅ Daily Summary saved to: data/output/daily_summary.csv")
        daily_summary_df.show(n=10, truncate=False)

        print("\n Inventory Levels are being updated...")
        updated_inventory_df = data_processor.update_inventory()
        data_processor.save_to_csv(
            updated_inventory_df, config["output_path"], "products_updated.csv"
        )

        data_processor.daily_summary_df = daily_summary_df

        # Generate forecasts
        try:
            # daily_summary_df follows the same schema as the daily_summary that you save to csv
            # schema:
            # - date: date - The business date
            # - num_orders: integer - Total number of orders for the day
            # - total_sales: decimal(10,2) - Total sales amount for the day
            # - total_profit: decimal(10,2) - Total profit for the day
            forecast_df = data_processor.forecast_sales_and_profits(
                data_processor.daily_summary_df
            )
            if forecast_df is not None:
                data_processor.save_to_csv(
                    forecast_df, config["output_path"], "sales_profit_forecast.csv"
                )
        except Exception as e:
            print(f"Warning: Could not generate forecasts: {str(e)}")
 

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise
    finally:
        print("\nCleaning up...")
        spark.stop()


if __name__ == "__main__":
    main()
