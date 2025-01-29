from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pyspark.sql.types import StringType

spark = SparkSession.builder.appName("TinyLlamaTopicClassification").getOrCreate()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Predefined topics
topics = ["Politics", "Technology", "Sports", "Education", "Health", "Environment"]

# Define UDF
def classify_topic(text):
    # Create candidate sentences by appending each topic as a prompt
    candidate_prompts = [f"{text} This text is about {topic}." for topic in topics]
    
    # Tokenize all candidate prompts together
    inputs = tokenizer(candidate_prompts, return_tensors="pt", padding=True, truncation=True)
    
    # Get classification scores for each prompt
    with torch.no_grad():
        outputs = model(**inputs).logits
        scores = torch.softmax(outputs, dim=1)[:, 1].tolist()  

    # Choose the topic with the highest confidence score
    best_topic = topics[scores.index(max(scores))]
    return best_topic

# Register the UDF
topic_classification_udf = udf(classify_topic, StringType())

# Example data
data = [
    (1, "The government passed a new law regarding taxation.", "Politics"),
    (2, "Apple releases its latest product with advanced features.", "Technology"),
    (3, "The football match ended with a stunning victory.", "Sports"),
    (4, "NASA announces new space exploration plans.", "Science"),
    (5, "The economy shows signs of improvement in the last quarter.", "Finance"),
    (6, "A new fitness trend is taking over social media.", "Health")
]

df = spark.createDataFrame(data, ["id", "text", "actual topic"])

# Apply the topic classification UDF
result_df = df.withColumn("predicted_topic", topic_classification_udf(df["text"]))
result_df.show(truncate=False)