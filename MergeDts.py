from pyspark.sql import SparkSession
from pyspark.sql.functions import *


spark = SparkSession.builder.appName("MergeDbs").getOrCreate()

df1 = spark.read.parquet("/user/s2853418/proj/train-00000-of-00018.parquet",
                         "/user/s2853418/proj/train-00001-of-00018.parquet",
                         "/user/s2853418/proj/train-00002-of-00018.parquet",
                         "/user/s2853418/proj/train-00003-of-00018.parquet",
                         "/user/s2853418/proj/train-00004-of-00018.parquet",
                         "/user/s2853418/proj/train-00005-of-00018.parquet",
                         "/user/s2853418/proj/train-00006-of-00018.parquet",
                         "/user/s2853418/proj/train-00007-of-00018.parquet",
                         "/user/s2853418/proj/train-00008-of-00018.parquet",
                         "/user/s2853418/proj/train-00009-of-00018.parquet",
                         "/user/s2853418/proj/train-00010-of-00018.parquet",
                         "/user/s2853418/proj/train-00011-of-00018.parquet",
                         "/user/s2853418/proj/train-00012-of-00018.parquet",
                         "/user/s2853418/proj/train-00013-of-00018.parquet",
                         "/user/s2853418/proj/train-00014-of-00018.parquet",
                         "/user/s2853418/proj/train-00015-of-00018.parquet",
                         "/user/s2853418/proj/train-00016-of-00018.parquet",
                         "/user/s2853418/proj/train-00017-of-00018.parquet")

df2 = spark.read.parquet("/user/s2853418/proj/train-00000-of-00006-4feeb3f83346a0e9.parquet",
                         "/user/s2853418/proj/train-00001-of-00006-4030672591c2f478.parquet",
                         "/user/s2853418/proj/train-00002-of-00006-1779b7cec9462180.parquet",
                         "/user/s2853418/proj/train-00003-of-00006-2fa862bfed56af1f.parquet",
                         "/user/s2853418/proj/train-00004-of-00006-18f4bdd50c103e71.parquet",
                         "/user/s2853418/proj/train-00005-of-00006-fe1acc5d10a9f0e2.parquet")

df1.printSchema()
df2.printSchema()

common_nested_fields = [
    "content", 
    "role"
]

# List the common fields for the nested 'conversation' column
common_nested_fields = ["content", "role"]

# Reconstruct the 'conversation' column with only common nested fields
# df1_trimmed = df1.withColumn(
#     "conversation",
#     expr(f"transform(conversation, x -> struct({','.join([f'x.{field}' for field in common_nested_fields])}))")
# )

df1_trimmed = df1.withColumn(
    "conversation",
    transform(
        col("conversation"),  
        lambda x: when(x.isNotNull(), struct(
            x["content"].alias("content"),
            x["role"].alias("role")
        )).otherwise(None)  # Explicitly allow null elements
    )
)

# Fix the extra fields in 'openai_moderation' by selecting only matching ones
df1_trimmed = df1_trimmed.withColumn(
    "openai_moderation",
    transform(
        col("openai_moderation"),
        lambda x: struct(
            struct(
                x["categories"]["harassment"].alias("harassment"),
                x["categories"]["harassment/threatening"].alias("harassment/threatening"),
                x["categories"]["hate"].alias("hate"),
                x["categories"]["hate/threatening"].alias("hate/threatening"),
                x["categories"]["self-harm"].alias("self-harm"),
                x["categories"]["self-harm/instructions"].alias("self-harm/instructions"),
                x["categories"]["self-harm/intent"].alias("self-harm/intent"),
                x["categories"]["sexual"].alias("sexual"),
                x["categories"]["sexual/minors"].alias("sexual/minors"),
                x["categories"]["violence"].alias("violence"),
                x["categories"]["violence/graphic"].alias("violence/graphic")
            ).alias("categories"),
            struct(
                x["category_scores"]["harassment"].alias("harassment"),
                x["category_scores"]["harassment/threatening"].alias("harassment/threatening"),
                x["category_scores"]["hate"].alias("hate"),
                x["category_scores"]["hate/threatening"].alias("hate/threatening"),
                x["category_scores"]["self-harm"].alias("self-harm"),
                x["category_scores"]["self-harm/instructions"].alias("self-harm/instructions"),
                x["category_scores"]["self-harm/intent"].alias("self-harm/intent"),
                x["category_scores"]["sexual"].alias("sexual"),
                x["category_scores"]["sexual/minors"].alias("sexual/minors"),
                x["category_scores"]["violence"].alias("violence"),
                x["category_scores"]["violence/graphic"].alias("violence/graphic")
            ).alias("category_scores"),
            x["flagged"].alias("flagged")
        )
    )
)


# Hardcode the columns present in df2 including the renamed identifier
columns_in_df2 = [
    "conversation",  
    "model",
    "turn",
    "language",
    "openai_moderation",
    "redacted",
    "conversation_id"  # Renamed column
]

# Rename the unique identifier column in df1
df1_trimmed = df1_trimmed.withColumnRenamed("conversation_hash", "conversation_id")

# Drop columns from df1 that are not in df2
df1_trimmed = df1_trimmed.drop(*[col for col in df1_trimmed.columns if col not in columns_in_df2])



# Verify schema consistency
print("Schema after removing columns and renaming df1:")
df1_trimmed.printSchema()
print("Schema of df2:")
df2.printSchema()

print(f"Row count of df1: {df1.count()}")
print(f"Row count of df1_trimmed: {df1_trimmed.count()}")
print(f"Row count of df2: {df2.count()}")

combined_df = df1_trimmed.unionByName(df2)
combined_df.printSchema()
print(f"Row count of combined_df: {combined_df.count()}")

# Save the combined DataFrame to HDFS as a Parquet file
combined_df.write.mode("overwrite").parquet("/user/s2853418/proj/combined_dataset.parquet")