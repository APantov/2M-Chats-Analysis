from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, sum as spark_sum
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.appName("CharacterCount").getOrCreate()

combined_df = spark.read.parquet("/user/s2853418/proj/combined_dataset.parquet")

# Counting characters in the conversation content
def extract_and_count_characters(conversation):
    if conversation:
        all_text = " ".join([item['content'] for item in conversation if 'content' in item])
        return len(all_text)
    return 0

count_characters_udf = udf(extract_and_count_characters, IntegerType())

# Adding character count column
combined_df_with_chars = combined_df.withColumn(
    "character_count", count_characters_udf(col("conversation"))
)

# Calculating total characters and estimating tokens
total_characters = combined_df_with_chars.select(spark_sum("character_count")).collect()[0][0]
estimated_tokens = total_characters / 4

# Display results
print(f"Total Characters: {total_characters}")
print(f"Estimated Tokens (1 token = 4 characters): {estimated_tokens}")
