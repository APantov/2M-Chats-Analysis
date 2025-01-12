from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, concat_ws
from pyspark.sql.types import StringType, FloatType
from nltk.sentiment.vader import SentimentIntensityAnalyzer

spark = SparkSession.builder.appName("SentimentAnalysisPerConversation").getOrCreate()
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

# Combine all messages into a single string for each conversation
combined_text_df = english_conversations_df.withColumn(
    "full_conversation_text", concat_ws(" ", col("conversation.content"))
)

# Add sentiment score for each message in the conversation
sentiment_df = combined_text_df.withColumn(
    "sentiment_score", sentiment_udf(col("full_conversation_text"))
)

sentiment_df.write.mode("overwrite").parquet("/user/s2853418/proj/sentiment_analysis_per_conversation.parquet")

sentiment_df.select("conversation.content", "sentiment_score").show(100)
