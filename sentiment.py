import dspy
dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class SentimentClassifier(dspy.Signature):
    """Classify the sentiment of a text."""

    text: str = dspy.InputField(desc="input text to classify sentiment")
    sentiment: int = dspy.OutputField(
        desc="sentiment, the higher the more positive", ge=0, le=10
    )

predict = dspy.Predict(SentimentClassifier) 

output = predict(text="I am feeling pretty happy!")
print(output)