import google.genai as genai
GEMINI_API_KEY='AIzaSyBXgl0ic3FYyPoCZobRs7LWGJEWA1kaO_A'
client = genai.Client(api_key=GEMINI_API_KEY)


models = client.models.list()

for model in models:
    print(model.name)
