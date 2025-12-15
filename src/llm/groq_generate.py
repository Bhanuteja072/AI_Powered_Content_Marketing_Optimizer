import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY missing in .env")

llm = ChatGroq(api_key=api_key, model_name="openai/gpt-oss-120b")
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template(
    """You are an expert digital marketer and copywriter.
    Write a professional and engaging post about {topic}

    Tone: {tone}
    Style: informative, clear, and audience-friendly.
    Use the following keywords naturally throughout the post:
    {keywords}
    
    Include 4-5 relevant hashtags. Prefer items from this provided list when applicable:
    {hashtags}
    If better, add other accurate, relevant hashtags (no filler or random tags). Do not exceed 8 total.


    Each post should:
    - Feel authentic and insightful.
    - Include 1 short call-to-action (CTA) or engaging line.
    - Avoid repetition or hashtags unless needed.
    Return a post suitable for LinkedIn or Twitter or youtube or any other platform around {max_words} words.
    """
)

chain = prompt | llm | parser


def generate_content(topic: str,tone: str ,max_words: int ,keywords: list[str] | None = None,hashtags: list[str] | None = None,) -> str:
    """
    Uses Groq model to generate short marketing posts.
    """
    keyword_text = ", ".join(keywords or [])
    hashtag_text = ", ".join(hashtags or [])

    return chain.invoke(
        {
            "topic": topic,
            "tone": tone,
            "keywords": keyword_text,
            "hashtags": hashtag_text,
            "max_words": max_words,
        }
    )