from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, lit, expr

spark = SparkSession.builder.appName("AverageSentimentAnalysisPerCategory").getOrCreate()

sentiment_df = spark.read.parquet("/user/s2853418/proj/sentiment_analysis_per_conversation.parquet")

# Check if ANY flagged or toxic category is True across the entire array
sentiment_df = sentiment_df.withColumn("flagged", expr("exists(openai_moderation, x -> x.flagged = true)")) \
                           .withColumn("harassment", expr("exists(openai_moderation, x -> x.categories.harassment = true)")) \
                           .withColumn("harassment_threatening", expr("exists(openai_moderation, x -> x.categories.`harassment/threatening` = true)")) \
                           .withColumn("hate", expr("exists(openai_moderation, x -> x.categories.hate = true)")) \
                           .withColumn("hate_threatening", expr("exists(openai_moderation, x -> x.categories.`hate/threatening` = true)")) \
                           .withColumn("self_harm", expr("exists(openai_moderation, x -> x.categories.`self-harm` = true)")) \
                           .withColumn("self_harm_instructions", expr("exists(openai_moderation, x -> x.categories.`self-harm/instructions` = true)")) \
                           .withColumn("self_harm_intent", expr("exists(openai_moderation, x -> x.categories.`self-harm/intent` = true)")) \
                           .withColumn("sexual", expr("exists(openai_moderation, x -> x.categories.sexual = true)")) \
                           .withColumn("sexual_minors", expr("exists(openai_moderation, x -> x.categories.`sexual/minors` = true)")) \
                           .withColumn("violence", expr("exists(openai_moderation, x -> x.categories.violence = true)")) \
                           .withColumn("violence_graphic", expr("exists(openai_moderation, x -> x.categories.`violence/graphic` = true)"))

# Avrg Sentiment for All Conversations
avg_sentiment_all = sentiment_df.agg(avg(col("sentiment_score")).alias("avg_sentiment")) \
                                           .withColumn("category", lit("All Conversations"))

# Avrg Sentiment for Toxic and Non-Toxic Conversations
avg_sentiment_toxic = sentiment_df.filter(col("flagged") == True) \
                                             .agg(avg(col("sentiment_score")).alias("avg_sentiment")) \
                                             .withColumn("category", lit("Toxic Conversations"))

avg_sentiment_non_toxic = sentiment_df.filter(col("flagged") == False) \
                                                 .agg(avg(col("sentiment_score")).alias("avg_sentiment")) \
                                                 .withColumn("category", lit("Non-Toxic Conversations"))

category_columns = [
    "harassment", "harassment_threatening", "hate", "hate_threatening", 
    "self_harm", "self_harm_instructions", "self_harm_intent",
    "sexual", "sexual_minors", "violence", "violence_graphic"
]

result_df = spark.createDataFrame([], schema=avg_sentiment_all.schema)

# avrg sentiment per toxic category
for category in category_columns:
    avg_sentiment_category = sentiment_df.filter(col(category) == True) \
                                                     .agg(avg(col("sentiment_score")).alias("avg_sentiment")) \
                                                     .withColumn("category", lit(category))
    result_df = result_df.union(avg_sentiment_category)

# All, Toxic, and Non-Toxic Conversations
result_df = result_df.union(avg_sentiment_all).union(avg_sentiment_toxic).union(avg_sentiment_non_toxic)

result_df.write.mode("overwrite").csv("/user/s2853418/average_sentiment_per_category_corrected.csv", header=True)

result_df.show(truncate=False)
