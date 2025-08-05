## Batch Processing ETL Pipeline for Candy Store

Tiger's Candy, a candy store that originated on the RIT campus, has rapidly gained popularity. To handle their growth, they have decided to build an automated data system for processing online orders in batches. 

This project implements a data processing pipeline for Tiger's Candy that processes sales data, updates inventory, and generates sales forecasts. The batch processing ETL pipeline integrates MySQL and MongoDB data, and Spark and uses Prophet for time series forecasting.

# Dataset Description 

The dataset contains customer, product, and transaction data from Tiger's Candy Store spanning February 1st to 10th, 2024.

- Customers (customers.csv)
It contains customer details: customer_id, first_name, last_name, email, address, phone.

- Products (products.csv)
It lists all the available products with attributes: product_id, product_name, category, subcategory, sales_price, cost_to_make, and stock.

- Raw Order Transactions (transactions_20240201.json - transactions_20240210.json)
It contains transaction details including transaction_id, customer_id, timestamp, and a list of ordered items (product_id, product_name, qty).

``` bash
  {
    "transaction_id": 73434473,
    "customer_id": 29,
    "timestamp": "2024-02-02T12:00:40.808092",
    "items": [
      {
        "product_id": 17,
        "product_name": "Sea Salt Crackle Enrobed Bites",
        "qty": 5
      },
      {
        "product_id": 18,
        "product_name": "Almond Shards Enrobed Bites",
        "qty": null
      },
      {
        "product_id": 3,
        "product_name": "Powdered Sugar Sticks Rectangles",
        "qty": 2
      }
    ]
  },
```

## Technologies Used
- Python
- MySQL (customer and product data)
- MongoDB (raw transaction data)
- Apache Spark (for distributed batch processing)
- Prophet (time series forecasting)

---

## Features 

### Data Loading: 
- Load customer and product data from CSV files into MySQL.
- Load transaction data from JSON files into MongoDB.
- Load MySQL data and MongoDB transaction data into a Spark session.

### Order Processing:
- Load and process orders from MySQL.
- Load and process order line items and remove invalid records.

### Daily Summary Calculation:
- Aggregate total sales and profit for each business date.
- Sort the fully canceled orders.

### Inventory Management:
- Deduct purchased quantities from stock.
- Ensure that the stock does not go below zero.
- Save updated inventory to **products_updated.csv**.

### Forecasting:
- Save the forecasted results to **sales_profit_forecast.csv**.

## Steps in the Pipeline

### 1. Data Initialization
- Load environment variables and configurations.
- Establish a Spark session.
- Initialize MySQL and MongoDB connections.

### 2. Data Loading & Preprocessing
- Read customers and products tables from MySQL.
- Load orders and order line items, ensuring valid entries.
- Join orders with products and customers for processing.

### 3. Processing Orders
- Sort and filter orders based on **order_datetime**.
- Remove fully canceled orders by checking **quantity** in **order_line_items**.
- Compute total sales and profit for valid orders.

### 4. Daily Summary Calculation
- Aggregate:
  - **num_orders**: Count distinct valid orders per day.
  - **total_sales**: Sum of valid line totals.
  - **total_profit**: Sum of calculated profit margins.
- Save to **daily_summary.csv**.

### 5. Inventory Update
- Deduct ordered quantities from stock.
- Ensure no negative stock values.
- Save updated inventory to **products_updated.csv**.

### 6. Forecasting Future Sales & Profits
- Use Prophet forecasting to model historical sales and profit trends.
- Save the predicted values into **sales_profit_forecast.csv**.

--- 

## Setup 

### Set Environment Variables

Create a `.env` file in the root directory by using the  `.env.example` file. 

### Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Pipeline

To run the pipeline,

```bash
python main.py
```

All the final CSVs will be saved in the path defined in `OUTPUT_PATH`.

---

## Output

The following output files are generated in the output/ directory:

**Processed Order Files**
+ batch_orders_YYYYMMDD.csv — Daily processed orders (e.g., batch_orders_20240201.csv)
+ batch_order_line_items_YYYYMMDD.csv — Line items for each day's batch

**Summary and Forecast Files**
- daily_summary.csv — Aggregated daily sales, profit, and order counts
- sales_profit_forecast.csv — Prophet forecast for upcoming sales and profits

**Final Datasets**
- orders.csv — All processed and valid orders
- order_line_items.csv — Valid order line items
- products_updated.csv — Final inventory after all deductions
---

## Formatting

This project uses [Black](https://black.readthedocs.io/en/stable/) for automatic code formatting.

To format the code:

```bash
black .
```

## Conclusion 

This batch ETL pipeline successfully automates the data processing workflow for Tiger's Candy. It integrates customer, product, and transaction data from MySQL and MongoDB, processes daily sales using Apache Spark, maintains inventory accuracy, and generates actionable insights through daily summaries and time series forecasting with Prophet.

**Key outcomes:**
- Cleaned and consolidated transactional data
- Accurate daily profit and sales tracking
- Inventory levels dynamically updated
- Data-driven forecasting for business planning

---

## Author
Bhavini Sai Mallu
Email: bhavini23sai@gmail.com
---
