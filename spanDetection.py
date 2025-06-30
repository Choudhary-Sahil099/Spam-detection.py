import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = {
    'email': [
        "Congratulations! You've won a free iPhone. Click here to claim now!",
        "Hey, are we still on for lunch today?",
        "Get rich quick with this secret investment opportunity!",
        "Don't forget to submit your assignment by tonight.",
        "URGENT: Your account has been compromised. Reset your password now!",
        "Meeting has been rescheduled to 3 PM tomorrow.",
        "You have been selected for a $500 gift card. Click here to win.",
        "Let's catch up soon!"
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam
}
df = pd.DataFrame(data)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['email'])
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

test_email = ["Win a free vacation to Maldives! Click now to claim."]
test_vector = vectorizer.transform(test_email)
prediction = model.predict(test_vector)

print("Spam" if prediction[0] == 1 else "Not Spam")