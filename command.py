from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

df = pd.DataFrame(data)
df = shuffle(df)

X = df["message"]
y = df["label"]

vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)