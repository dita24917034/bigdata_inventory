import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, round as spark_round, to_date, month, year, concat_ws
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Inisialisasi Spark
spark = SparkSession.builder.appName("ProfitMarginAnalysis").getOrCreate()

st.title("📊 Analisis & Prediksi Profit Margin")

# ======= Membaca file lokal ========
purchasing_file = "data/PurchaseFINAL12312016.csv"
sales_file = "data/salesFINAL12312016.csv"
price_file = "data/2017PurchasePricesDec.csv"

df_purchase = spark.read.csv(purchasing_file, header=True, inferSchema=True)
df_sales = spark.read.csv(sales_file, header=True, inferSchema=True)
df_price = spark.read.csv(price_file, header=True, inferSchema=True)

# ======= Filter kolom dan gabungkan data ========
df_sales = df_sales.withColumn("salesdate", to_date(col("salesdate"), "yyyy-MM-dd"))
df_sales = df_sales.withColumn("periode", concat_ws("-", year("salesdate"), month("salesdate")))

df_purchase = df_purchase.withColumn("receivingdate", to_date(col("receivingdate"), "yyyy-MM-dd"))
df_purchase = df_purchase.withColumn("periode", concat_ws("-", year("receivingdate"), month("receivingdate")))

df_price = df_price.select("brand", "description", "size", "purchaseprice")

# Gabungkan data purchasing dan price
df_joined = df_purchase.join(df_price, on=["brand", "description", "size"], how="left")

# Gabungkan dengan data sales
df_final = df_joined.join(df_sales, on=["inventoryid", "storeid", "brand", "description", "size"], how="inner")

# Hitung total cost & total revenue
df_final = df_final.withColumn("total_cost", col("salesquantity") * col("purchaseprice"))
df_final = df_final.withColumn("total_revenue", col("salesquantity") * col("salesprice"))

# Hitung profit margin per periode
df_margin = df_final.groupBy("periode").agg(
    spark_sum("total_cost").alias("total_cost"),
    spark_sum("total_revenue").alias("total_revenue")
)

df_margin = df_margin.withColumn("profit_margin", spark_round(
    (col("total_revenue") - col("total_cost")) / col("total_revenue") * 100, 2
))

# ======= Konversi ke pandas untuk visualisasi & ML ========
df_plot = df_margin.orderBy("periode").toPandas()

# ======= Filter by brand (opsional) ========
brands = df_final.select("brand").distinct().toPandas()["brand"].dropna().unique()
selected_brand = st.selectbox("🧵 Filter Brand", options=brands)

if selected_brand:
    df_filtered = df_final.filter(col("brand") == selected_brand)
    df_margin_filtered = df_filtered.groupBy("periode").agg(
        spark_sum("total_cost").alias("total_cost"),
        spark_sum("total_revenue").alias("total_revenue")
    )
    df_margin_filtered = df_margin_filtered.withColumn("profit_margin", spark_round(
        (col("total_revenue") - col("total_cost")) / col("total_revenue") * 100, 2
    ))
    df_plot = df_margin_filtered.orderBy("periode").toPandas()

# ======= Visualisasi profit margin ========
st.subheader("📈 Profit Margin per Periode")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_plot["periode"], df_plot["profit_margin"], marker='o', color='blue')
ax.set_title("Profit Margin per Periode")
ax.set_xlabel("Periode")
ax.set_ylabel("Profit Margin (%)")
ax.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig)

# ======= Model ML: Prediksi Profit Margin ========
st.subheader("🤖 Prediksi Profit Margin dengan Random Forest")

if len(df_plot) >= 2:
    df_plot["period_num"] = np.arange(len(df_plot))  # fitur numerik
    X = df_plot[["period_num"]]
    y = df_plot["profit_margin"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)

    df_plot["predicted_margin"] = y_pred
    mse = mean_squared_error(y, y_pred)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(df_plot["periode"], df_plot["profit_margin"], label="Aktual", marker='o', color='blue')
    ax2.plot(df_plot["periode"], df_plot["predicted_margin"], label="Prediksi", linestyle='--', color='orange')
    ax2.set_title("Prediksi Profit Margin per Periode")
    ax2.set_xlabel("Periode")
    ax2.set_ylabel("Profit Margin (%)")
    ax2.legend()
    ax2.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.markdown(f"📉 **Mean Squared Error (MSE):** `{mse:.2f}`")
else:
    st.warning("Data belum cukup untuk prediksi. Tambahkan lebih banyak periode.")
