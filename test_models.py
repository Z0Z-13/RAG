from google import genai

client = genai.Client(api_key="AIzaSyBki14yGQOcknSzSgoA5vL9VgY-ypSVeeI")

models = client.models.list()

for m in models:
    print(m.name)