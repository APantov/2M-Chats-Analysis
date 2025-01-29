from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pyspark.sql.types import StringType

spark = SparkSession.builder.appName("TopicClassification").getOrCreate()

# Predefined topics
topics = ["Politics", "Technology", "Sports", "Education", "Health", "Environment", "Finance", "Entertainment", "Science", "Travel"]

# Load a small model
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSequenceClassification.from_pretrained("google-t5/t5-small").half()

def classify_topic(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs["input_ids"] = inputs["input_ids"].to(torch.long)    
    with torch.no_grad():
        outputs = model(**inputs).logits
    scores = torch.softmax(outputs, dim=1).squeeze().tolist()
    best_topic_index = scores.index(max(scores))
    return topics[best_topic_index]

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
