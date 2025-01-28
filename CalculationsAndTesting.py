from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, mean, variance, explode
from scipy import stats
import numpy as np

spark = SparkSession.builder.appName("CharacterCount").getOrCreate()

#downloads file from specified path
combined_df = spark.read.parquet("/user/s2853418/proj/sentiment_eng.parquet")

#filters the dataframe and takes a specific categorie
filtered_df = combined_df.filter(F.expr("exists(openai_moderation, x -> x.categories.hate = true)"))

#explodes the dataframe to acess element "openai_moderation" in column array
exploded_df = filtered_df.withColumn("element", explode(col("openai_moderation")))

#retrieve non_toxic statistics such as mean and variance
non_toxic_stats = (
    exploded_df.filter(col("element.flagged") == False)
    .select("sentiment_score")
    .agg(
        mean("sentiment_score").alias("mean_non_toxic"),
        variance("sentiment_score").alias("variance_non_toxic")
    )
)

#retrive toxic statistics
toxic_stats = (
    exploded_df.filter(col("element.flagged") == True)
    .select("sentiment_score")
    .agg(
        mean("sentiment_score").alias("mean_toxic"),
        variance("sentiment_score").alias("variance_toxic")
    )
)

#retrive the total number of flagged and non_flagged instances
flagged_counts = (
    exploded_df.groupBy(col("element.flagged").alias("is_flagged"))
    .count()
    .withColumnRenamed("count", "conversation_count")
)

#show the tables
non_toxic_stats.show()
toxic_stats.show()
flagged_counts.show()


#calculated t_value based on the tables
t_value = 61.04519
df = 387304.11814

# Calculate the p-value for a two-tailed Welch's t-test
p_value = 2 * (1 - stats.t.cdf(t_value, df))
print(f"P-value: {p_value}")

#mean and variance values from the table
mean_toxic = 0.16798251813563223
mean_non_toxic = 0.2422724576509059
variance_non_toxic = 0.3216997731157445
variance_toxic = 0.41499796584208665

#number of isntances from the tables
n_toxic = 312338
n_non_toxic = 2111932


#calculate pooled std for cohen'd value
pooled_std = np.sqrt(((n_toxic - 1) * variance_toxic + (n_non_toxic - 1) * variance_non_toxic) / (n_toxic + n_non_toxic - 2))

cohen_d = (mean_toxic - mean_non_toxic) / pooled_std

print(f"Cohen's d: {cohen_d}")