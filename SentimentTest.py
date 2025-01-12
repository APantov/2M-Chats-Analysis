from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Spark session
spark = SparkSession.builder.appName("SentimentAnalysisTest").getOrCreate()

# Sample data
data = [
    (1, "I love the new design of your website!"),
    (2, "The service was terrible and I'm not happy."),
    (3, "I'm feeling neutral about the recent updates."),
    (4, "Absolutely fantastic experience, will come again!"),
    (5, "Not what I expected, quite disappointing.")
]

# Create DataFrame
columns = ["id", "text"]
df = spark.createDataFrame(data, columns)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define UDF to compute sentiment
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    return scores['compound']

# Register UDF
sentiment_udf = udf(get_sentiment, StringType())

# Apply UDF to DataFrame
result_df = df.withColumn("sentiment_score", sentiment_udf(df["text"]))

# Show results
result_df.show()

