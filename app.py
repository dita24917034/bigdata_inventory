import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, date_format, sum
import pandas as pd
import matplotlib.pyplot as plt

# Inisialisasi SparkSession
spark = SparkSession.builder \
    .appName("ProfitMarginStreamlit") \
    .getOrCreate()

# Judul Aplikasi
st.title("📊 Analisis Profit Margin Berdasarkan Brand (Tanpa Upload)")

# --- 🔹 BACA FILE LANGSUNG DARI LOKAL PATH ---
# Pastikan path file sesuai lokasi di folder kamu
purchasing_path = "data/PurchasesFINAL12312016.csv"
sales_path = "data/salePurchasesFINAL12312016s.csv"
price_path = "data/2017PurchasePricesDec.csv"

# Baca CSV ke DataFrame PySpark
purchasing_df = spark.read.option("header", True).option("inferSchema", True).csv(purchasing_path)
sales_df = spark.read.option("header", True).option("inferSchema", True).csv(sales_path)
price_df = spark.read.option("header", True).option("inferSchema", True).csv(price_path)

# Konversi kolom tanggal
purchasing_df = purchasing_df.withColumn("podate", to_date("podate", "yyyy-MM-dd"))
sales_df = sales_df.withColumn("salesdate", to_date("salesdate", "yyyy-MM-dd"))

# Tambahkan kolom 'period' (bulan & tahun)
purchasing_df = purchasing_df.withColumn("period", date_format("podate", "yyyy-MM"))
sales_df = sales_df.withColumn("period", date_format("salesdate", "yyyy-MM"))

# Gabungkan harga pembelian ke data purchasing
purchasing_with_price = purchasing_df.join(
    price_df,
    on=["brand", "description", "size", "vendornumber", "vendorname"],
    how="left"
)

# Hitung total cost (asumsikan 1 baris = 1 unit jika tidak ada qty)
purchasing_with_price = purchasing_with_price.withColumn("total_cost", col("purchaseprice"))

# Ambil daftar brand unik
all_brands = purchasing_with_price.select("brand").distinct().orderBy("brand").toPandas()["brand"].dropna().tolist()

# Dropdown untuk memilih brand
selected_brand = st.selectbox("🔍 Pilih Brand", options=all_brands)

# Filter data berdasarkan brand
filtered_purchase = purchasing_with_price.filter(col("brand") == selected_brand)
filtered_sales = sales_df.filter(col("brand") == selected_brand)

# Ringkasan pembelian per bulan
purchase_summary = filtered_purchase.groupBy("period") \
    .agg(sum("total_cost").alias("total_purchase_cost"))

# Ringkasan penjualan per bulan
sales_summary = filtered_sales.groupBy("period") \
    .agg(
        sum("salesdollars").alias("total_sales_dollars"),
        sum("salesquantity").alias("total_sales_qty")
    )

# Gabungkan & hitung profit margin
combined = sales_summary.join(purchase_summary, on="period", how="left") \
    .fillna(0) \
    .withColumn("profit", col("total_sales_dollars") - col("total_purchase_cost")) \
    .withColumn("profit_margin", (col("profit") / col("total_sales_dollars")) * 100)

# Konversi ke Pandas
df_plot = combined.select("period", "profit_margin").orderBy("period").toPandas()

# Visualisasi Line Chart
st.subheader(f"📈 Grafik Profit Margin - Brand: {selected_brand}")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_plot["period"], df_plot["profit_margin"], marker="o", color="blue")
ax.set_xlabel("Periode (YYYY-MM)")
ax.set_ylabel("Profit Margin (%)")
ax.set_title(f"Profit Margin Bulanan untuk {selected_brand}")
ax.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig)

# Tampilkan Dataframe
with st.expander("📋 Lihat Data Tabel"):
    st.dataframe(df_plot)
