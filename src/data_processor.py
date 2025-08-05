from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    explode,
    col,
    round as spark_round,
    sum as spark_sum,
    count,
    abs as spark_abs,
    when,
    to_date,
    countDistinct,
    date_format,
)
from typing import Dict, Tuple
import os
import glob
import shutil
import decimal
import numpy as np
from time_series import ProphetForecaster
from datetime import datetime, timedelta
from pyspark.sql.types import DoubleType, DecimalType
from time_series import ProphetForecaster


class DataProcessor:

    def __init__(self, spark: SparkSession):
        self.spark = spark
        # Initialize all class properties
        self.config = None
        self.current_inventory = None
        self.inventory_initialized = False
        self.original_products_df = None  # Store original products data
        self.reload_inventory_daily = False  # New flag for inventory reload
        self.order_items = None
        self.products_df = None
        self.customers_df = None
        self.transactions_df = None
        self.orders_df = None
        self.order_line_items_df = None
        self.daily_summary_df = None
        self.total_cancelled_items = 0

    def configure(self, config: Dict) -> None:
        """Configure the data processor with environment settings"""
        self.config = config
        self.reload_inventory_daily = config.get("reload_inventory_daily", False)
        print("\nINITIALIZING DATA SOURCES")
        print("-" * 80)
        if self.reload_inventory_daily:
            print("Daily inventory reload: ENABLED")
        else:
            print("Daily inventory reload: DISABLED")

    def finalize_processing(self) -> None:
        """Finalize processing and create summary"""
        print("\nPROCESSING COMPLETE")
        print("=" * 80)
        print(f"Total Cancelled Items: {self.total_cancelled_items}")

    # ------------------------------------------------------------------------------------------------
    # Try not to change the logic of the time series forecasting model
    # DO NOT change functions with prefix _
    # ------------------------------------------------------------------------------------------------
    def forecast_sales_and_profits(
        self, daily_summary_df: DataFrame, forecast_days: int = 1
    ) -> DataFrame:
        """
        Main forecasting function that coordinates the forecasting process
        """
        try:
            # Build model
            model_data = self.build_time_series_model(daily_summary_df)

            # Calculate accuracy metrics
            metrics = self.calculate_forecast_metrics(model_data)

            # Generate forecasts
            forecast_df = self.make_forecasts(model_data, forecast_days)

            return forecast_df

        except Exception as e:
            print(
                f"Error in forecast_sales_and_profits: {str(e)}, please check the data"
            )
            return None

    def print_inventory_levels(self) -> None:
        """Print current inventory levels for all products"""
        print("\nCURRENT INVENTORY LEVELS")
        print("-" * 40)

        inventory_data = self.current_inventory.orderBy("product_id").collect()
        for row in inventory_data:
            print(
                f"• {row['product_name']:<30} (ID: {row['product_id']:>3}): {row['current_stock']:>4} units"
            )
        print("-" * 40)

    def build_time_series_model(self, daily_summary_df: DataFrame) -> dict:
        """Build Prophet models for sales and profits"""
        print("\n" + "=" * 80)
        print("TIME SERIES MODEL CONSTRUCTION")
        print("-" * 80)

        model_data = self._prepare_time_series_data(daily_summary_df)
        return self._fit_forecasting_models(model_data)

    def calculate_forecast_metrics(self, model_data: dict) -> dict:
        """Calculate forecast accuracy metrics for both models"""
        print("\nCalculating forecast accuracy metrics...")

        # Get metrics from each model
        sales_metrics = model_data["sales_model"].get_metrics()
        profit_metrics = model_data["profit_model"].get_metrics()

        metrics = {
            "sales_mae": sales_metrics["mae"],
            "sales_mse": sales_metrics["mse"],
            "profit_mae": profit_metrics["mae"],
            "profit_mse": profit_metrics["mse"],
        }

        # Print metrics and model types
        print("\nForecast Error Metrics:")
        print(f"Sales Model Type: {sales_metrics['model_type']}")
        print(f"Sales MAE: ${metrics['sales_mae']:.2f}")
        print(f"Sales MSE: ${metrics['sales_mse']:.2f}")
        print(f"Profit Model Type: {profit_metrics['model_type']}")
        print(f"Profit MAE: ${metrics['profit_mae']:.2f}")
        print(f"Profit MSE: ${metrics['profit_mse']:.2f}")

        return metrics

    def make_forecasts(self, model_data: dict, forecast_days: int = 7) -> DataFrame:
        """Generate forecasts using Prophet models"""
        print(f"\nGenerating {forecast_days}-day forecast...")

        forecasts = self._generate_model_forecasts(model_data, forecast_days)
        forecast_dates = self._generate_forecast_dates(
            model_data["training_data"]["dates"][-1], forecast_days
        )

        return self._create_forecast_dataframe(forecast_dates, forecasts)

    def _prepare_time_series_data(self, daily_summary_df: DataFrame) -> dict:
        """Prepare data for time series modeling"""
        data = (
            daily_summary_df.select("date", "total_sales", "total_profit")
            .orderBy("date")
            .collect()
        )

        dates = np.array([row["date"] for row in data])
        sales_series = np.array([float(row["total_sales"]) for row in data])
        profit_series = np.array([float(row["total_profit"]) for row in data])

        self._print_dataset_info(dates, sales_series, profit_series)

        return {"dates": dates, "sales": sales_series, "profits": profit_series}

    def _print_dataset_info(
        self, dates: np.ndarray, sales: np.ndarray, profits: np.ndarray
    ) -> None:
        """Print time series dataset information"""
        print("Dataset Information:")
        print(f"• Time Period:          {dates[0]} to {dates[-1]}")
        print(f"• Number of Data Points: {len(dates)}")
        print(f"• Average Daily Sales:   ${np.mean(sales):.2f}")
        print(f"• Average Daily Profit:  ${np.mean(profits):.2f}")

    def _fit_forecasting_models(self, data: dict) -> dict:
        """Fit Prophet models to the prepared data"""
        print("\nFitting Models...")
        sales_forecaster = ProphetForecaster()
        profit_forecaster = ProphetForecaster()

        sales_forecaster.fit(data["sales"])
        profit_forecaster.fit(data["profits"])
        print("Model fitting completed successfully")
        print("=" * 80)

        return {
            "sales_model": sales_forecaster,
            "profit_model": profit_forecaster,
            "training_data": data,
        }

    def _generate_model_forecasts(self, model_data: dict, forecast_days: int) -> dict:
        """Generate forecasts from both models"""
        return {
            "sales": model_data["sales_model"].predict(forecast_days),
            "profits": model_data["profit_model"].predict(forecast_days),
        }

    def _generate_forecast_dates(self, last_date: datetime, forecast_days: int) -> list:
        """Generate dates for the forecast period"""
        return [last_date + timedelta(days=i + 1) for i in range(forecast_days)]

    def _create_forecast_dataframe(self, dates: list, forecasts: dict) -> DataFrame:
        """Create Spark DataFrame from forecast data"""
        forecast_rows = [
            (date, float(sales), float(profits))
            for date, sales, profits in zip(
                dates, forecasts["sales"], forecasts["profits"]
            )
        ]

        return self.spark.createDataFrame(
            forecast_rows, ["date", "forecasted_sales", "forecasted_profit"]
        )

    def load_mysql_data(
        self, jdbc_url: str, db_table: str, db_user: str, db_password: str
    ) -> DataFrame:
        return (
            self.spark.read.format("jdbc")
            .option("url", jdbc_url)
            .option("driver", "com.mysql.cj.jdbc.Driver")
            .option("dbtable", db_table)
            .option("user", db_user)
            .option("password", db_password)
            .load()
        )

    def load_mongo_data(self, db_name: str, collection_name: str) -> DataFrame:
        return (
            self.spark.read.format("mongo")
            .option("database", db_name)
            .option("collection", collection_name)
            .load()
        )

    def load_and_process_orders(self) -> DataFrame:
        print("\n Loading and Processing Orders...")

        orders_df = self.load_mysql_data(
            self.config["mysql_url"],
            "orders",
            self.config["mysql_user"],
            self.config["mysql_password"],
        )

        orders_df = orders_df.withColumn(
            "order_datetime",
            date_format(col("order_datetime"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"),
        )

        # sorting orders by order_id
        sorted_orders_df = orders_df.orderBy("order_id")

        print(f"\n Orders Processed: {sorted_orders_df.count()} Rows")
        sorted_orders_df.show(n=5, truncate=False)

        return sorted_orders_df

    def load_and_process_order_line_items(self) -> DataFrame:
        print("\n Loading and Processing Order Line Items...")

        order_items_df = self.load_mysql_data(
            self.config["mysql_url"],
            "order_line_items",
            self.config["mysql_user"],
            self.config["mysql_password"],
        )

        # removing rows where quantity is null
        order_items_clean_df = order_items_df.filter(col("quantity").isNotNull())

        # sorting order line items
        sorted_order_items_df = order_items_clean_df.orderBy("order_id", "product_id")

        print(f"\n Order Line Items Processed: {sorted_order_items_df.count()} Rows")
        sorted_order_items_df.show(n=5, truncate=False)

        return sorted_order_items_df

    def calculate_daily_summary(self, orders_df, order_items_df, product_df):
        """Calculate daily summary from processed data"""
        print("\n Creating Daily Summary Table...")

        orders_df = orders_df.orderBy("order_datetime")
        orders_items_joined = orders_df.join(order_items_df, "order_id", "inner")
        orders_items_joined = orders_items_joined.join(product_df, "product_id", "left")

        # Filtering out invalid quantities (null or zero)
        orders_items_joined = orders_items_joined.filter(
            (col("quantity").isNotNull()) & (col("quantity") > 0)
        )

        # ✅ Now compute profit
        orders_items_joined = orders_items_joined.withColumn(
            "profit",
            spark_round((col("unit_price") - col("cost_to_make")) * col("quantity"), 2),
        )

        # canceled orders
        canceled_orders_df = orders_items_joined.groupBy("order_id").agg(
            spark_sum("quantity").alias("total_qty")
        )
        canceled_orders_df = canceled_orders_df.filter(col("total_qty") == 0)

        # valid orders
        valid_orders_df = orders_items_joined.groupBy("order_id").agg(
            spark_sum("quantity").alias("total_qty")
        )
        valid_orders_df = valid_orders_df.filter(col("total_qty") > 0)

        orders_filtered = orders_items_joined.join(valid_orders_df, "order_id", "inner")

        # finding daily summary
        daily_summary_df = (
            orders_filtered.withColumn("date", to_date(col("order_datetime")))
            .groupBy("date")
            .agg(
                countDistinct("order_id").alias("num_orders"),
                spark_round(spark_sum("line_total"), 2).alias("total_sales"),
                spark_round(spark_sum("profit"), 2).alias("total_profit"),
            )
            .orderBy("date")
        )

        return daily_summary_df

    def update_inventory(self):
        print("\n Updating Product Inventory...")

        # loading the products and order line items
        products_df = self.load_mysql_data(
            self.config["mysql_url"],
            self.config["products_table"],
            self.config["mysql_user"],
            self.config["mysql_password"],
        )

        order_items_df = self.load_mysql_data(
            self.config["mysql_url"],
            "order_line_items",
            self.config["mysql_user"],
            self.config["mysql_password"],
        )

        order_summary_df = order_items_df.groupBy("product_id").agg(
            spark_sum("quantity").alias("total_ordered")
        )

        # joining products with order line items on product_id
        inventory_update_df = products_df.join(order_summary_df, "product_id", "left")

        # reducing ordered quantity from stock
        inventory_update_df = inventory_update_df.withColumn(
            "current_stock",
            when(
                col("total_ordered").isNotNull(), col("stock") - col("total_ordered")
            ).otherwise(col("stock")),
        )

        # stock should not be less than 0
        inventory_update_df = inventory_update_df.withColumn(
            "current_stock",
            when(col("current_stock") < 0, 0).otherwise(col("current_stock")),
        )

        # required columns selected and sort by product_id
        products_updated_df = inventory_update_df.select(
            "product_id", "product_name", "current_stock"
        ).orderBy("product_id")

        # updated inventory is saved to MySQL table 'products_updated'
        products_updated_df.write.format("jdbc").options(
            url=self.config["mysql_url"],
            driver="com.mysql.cj.jdbc.Driver",
            dbtable="products_updated",
            user=self.config["mysql_user"],
            password=self.config["mysql_password"],
        ).mode("overwrite").save()

        print("\n Inventory is updated and saved to 'products_updated' table.")
        products_updated_df.show(n=10, truncate=False)

        return products_updated_df

    def save_to_csv(self, df: DataFrame, output_path: str, filename: str):
        """Save DataFrame to a CSV file in the specified output path."""
        temp_dir = os.path.join(output_path, f"temp_{filename}")
        final_path = os.path.join(output_path, filename)

        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(temp_dir)

        temp_files = glob.glob(os.path.join(temp_dir, "part-*.csv"))
        if temp_files:
            shutil.move(temp_files[0], final_path)

            shutil.rmtree(temp_dir)
            print(f"\n {filename} saved to: {final_path}")


def process_day_transactions(spark, config, date_str):
    print(f"\n📦 Processing transactions for {date_str}")
    collection_name = f"{config['mongodb_collection_prefix']}_{date_str}"
    transactions_df = (
        spark.read.format("mongo")
        .option("uri", config["mongodb_uri"])
        .option("database", config["mongodb_db"])
        .option("collection", collection_name)
        .load()
    )

    print(f"📂 Checking collection: {collection_name}")
    transactions_df.printSchema()
    transactions_df.show(3, truncate=False)

    if "items" not in transactions_df.columns:
        print(f"⚠️  Skipping {collection_name}: 'items' field missing.")
        return spark.createDataFrame(
            [], schema="order_id INT, customer_id INT, order_datetime STRING"
        ), spark.createDataFrame(
            [],
            schema="order_id INT, product_id INT, quantity INT, unit_price DECIMAL(10,2), line_total DECIMAL(10,2)",
        )

    flat_items_df = (
        transactions_df.withColumn("item", explode("items"))
        .select(
            col("transaction_id").alias("order_id"),
            col("customer_id"),
            col("timestamp").alias("order_datetime"),
            col("item.product_id"),
            col("item.product_name"),
            col("item.qty").alias("quantity"),
        )
        .filter(col("quantity").isNotNull())
    )
    products_df = (
        spark.read.format("jdbc")
        .option("url", config["mysql_url"])
        .option("driver", "com.mysql.cj.jdbc.Driver")
        .option("dbtable", config["products_table"])
        .option("user", config["mysql_user"])
        .option("password", config["mysql_password"])
        .load()
    )
    items_joined = flat_items_df.join(products_df, "product_id", "left")
    items_validated = (
        items_joined.withColumn(
            "final_quantity",
            when(col("quantity") <= col("stock"), col("quantity")).otherwise(None),
        )
        .withColumn(
            "line_total", spark_round(col("final_quantity") * col("sales_price"), 2)
        )
        .withColumnRenamed("sales_price", "unit_price")
    )
    valid_orders_df = (
        items_validated.filter(col("final_quantity").isNotNull())
        .groupBy("order_id")
        .agg(spark_sum("final_quantity").alias("total_qty"))
        .filter(col("total_qty") > 0)
    )
    orders_df = (
        items_validated.join(valid_orders_df, "order_id", "inner")
        .select("order_id", "customer_id", "order_datetime")
        .distinct()
    )
    order_items_df = items_validated.select(
        "order_id",
        "product_id",
        col("final_quantity").cast("int").alias("quantity"),
        col("unit_price").cast(DecimalType(10, 2)),
        col("line_total").cast(DecimalType(10, 2)),
    ).orderBy("order_id", "product_id")
    return orders_df, order_items_df


def save_to_csv(df, output_path, filename):
    temp_dir = os.path.join(output_path, f"temp_{filename}")
    final_path = os.path.join(output_path, filename)
    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(temp_dir)
    csv_file = glob.glob(os.path.join(temp_dir, "part-*.csv"))[0]
    shutil.move(csv_file, final_path)
    shutil.rmtree(temp_dir)
    print(f"✅ Saved {filename}")
