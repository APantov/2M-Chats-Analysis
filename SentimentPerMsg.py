from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, explode
from pyspark.sql.types import StringType, FloatType
from nltk.sentiment.vader import SentimentIntensityAnalyzer

spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
analyzer = SentimentIntensityAnalyzer()

combined_df = spark.read.parquet("/user/s2853418/proj/combined_dataset.parquet")

# Only_english convos
english_conversations_df = combined_df.filter(col("language") == "English")

# UDF for sentiment analysis
def get_sentiment_score(text):
    if text:
        sentiment = analyzer.polarity_scores(text)
        return sentiment['compound']
    else:
        return None

sentiment_udf = udf(get_sentiment_score, FloatType())

# Explode conversation array to process each message individually
exploded_df = english_conversations_df.withColumn("conversation", explode(col("conversation")))

# Add sentiment score for each message in the conversation
sentiment_df = exploded_df.withColumn("sentiment_score", sentiment_udf(col("conversation.content")))

sentiment_df.write.mode("overwrite").parquet("/user/s2853418/proj/sentiment_eng.parquet")

sentiment_df.select("conversation.content", "sentiment_score").show(100)
