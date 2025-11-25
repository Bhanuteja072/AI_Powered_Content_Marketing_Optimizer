import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY missing in .env")

llm=ChatGroq(model="llama-3.3-70b-versatile")


def main():
    try:
        prompt = "Generate a catchy title for an article about AI-powered content marketing."
        response = llm.invoke(prompt)
        print("Generated Title:", response.content)
    except Exception as e:
        print("Error:", repr(e))

if __name__ == "__main__":
    main()