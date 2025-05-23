import os
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import col, count, lower, regexp_replace, lit, to_timestamp
from pyspark.sql.types import BooleanType


class TwitterDataTransformer:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.spark = self._create_spark_session()

    def _create_spark_session(self) -> SparkSession:
        return SparkSession.builder \
            .appName("Twitter Data Transformation") \
            .config("spark.driver.memory", "8g") \
            .config("spark.sql.shuffle.partitions", "8") \
            .config("spark.default.parallelism", "8") \
            .config("spark.hadoop.fs.file.impl.disable.cache", "true") \
            .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem") \
            .config("spark.hadoop.fs.checksum.write", "false") \
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
            .getOrCreate()

    def load_data(self) -> DataFrame:
        df = self.spark.read.option("header", True).option("multiLine", True).csv(self.input_path)
        print("Number of lines:", df.count())
        return df

    def transform(self, df: DataFrame) -> DataFrame:
        #inbound = True → The tweet is incoming, from a customer to the brand.
        #inbound = False → The tweet is outgoing, from the brand to a customer.
        df = df.withColumn("inbound", col("inbound").cast(BooleanType())) \
               .withColumn("created_at", to_timestamp(col("created_at"), "EEE MMM dd HH:mm:ss Z yyyy")) \
               .orderBy("created_at")

        df = df.withColumn("clean_text", lower(col("text")))\
                .withColumn("clean_text", regexp_replace("clean_text", r"http\S+", ""))\
                .withColumn("clean_text", regexp_replace("clean_text", r"[^a-zA-Z0-9\s@]", ""))\
                .drop("text")

        window = Window.partitionBy("author_id")
        author_tweet_counts = df.withColumn("tweet_count", count("*").over(window))

        brand_authors = author_tweet_counts.filter((col("inbound") == False) & (col("tweet_count") > 5)) \
                                           .select("author_id").distinct()

        df = df.join(brand_authors.withColumn("is_brand", lit(True)), on="author_id", how="left") \
               .fillna(False, subset=["is_brand"])

        df = df.filter(col("inbound") == True)
        return df

    def write_output(self, df: DataFrame):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        try:
            df.write.mode("overwrite").parquet(self.output_path)
            print("Parquet written successfully")
        except Exception as e:
            print("Error writing Parquet:", e)
            raise

    def run(self):
        try:
            df_raw = self.load_data()
            df_transformed = self.transform(df_raw)
            self.write_output(df_transformed)
        finally:
            self.spark.stop()


if __name__ == "__main__":
    INPUT_PATH = "data/raw/customer_support_twitter/twcs/twcs.csv"
    OUTPUT_PATH = "data/processed/curated/"

    transformer = TwitterDataTransformer(INPUT_PATH, OUTPUT_PATH)
    transformer.run()