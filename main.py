import streamlit as st
import requests
from newspaper import Article, Config
from urllib.parse import quote
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from heapq import nlargest
import time
import json
from duckduckgo_search import DDGS
from datetime import datetime
from deep_translator import GoogleTranslator
from deep_translator.exceptions import RequestError
import re
import unicodedata
from transformers import pipeline

# Set page config
st.set_page_config(
    page_title="AI News Hub",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with modern design
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1b26 0%, #13151f 100%);
    }
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #ff6b6b 0%, #ff5252 100%);
        color: white;
        border: none;
        padding: 0.75rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 82, 82, 0.2);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff5252 0%, #ff4242 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 82, 82, 0.3);
    }
    .article-card {
        background: rgba(30, 32, 47, 0.95);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    .article-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        border-color: rgba(82, 183, 255, 0.3);
    }
    .article-title {
        color: #ffffff;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        line-height: 1.4;
    }
    .article-summary {
        color: #a0aec0;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    .article-meta {
        color: #718096;
        font-size: 0.875rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .source-tag {
        background: rgba(82, 183, 255, 0.1);
        color: #52b7ff;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        border: 1px solid rgba(82, 183, 255, 0.2);
    }
    .header-container {
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #52b7ff 0%, #1e90ff 100%);
        color: white;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(82, 183, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at top right, rgba(255,255,255,0.1) 0%, transparent 60%);
        pointer-events: none;
    }
    .header-container h1 {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .header-container p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .sidebar-content {
        background: rgba(30, 32, 47, 0.95);
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    /* Style for input fields */
    .stTextInput > div > div {
        background: rgba(30, 32, 47, 0.95) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 25px !important;
        color: white !important;
        padding: 0.5rem 1rem !important;
    }
    .stTextInput > div > div:focus-within {
        border-color: #52b7ff !important;
        box-shadow: 0 0 0 1px #52b7ff !important;
    }
    .stTextInput > div > div > input {
        color: white !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #718096 !important;
    }
    /* Style for select boxes */
    .stSelectbox > div > div {
        background: rgba(30, 32, 47, 0.95) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 25px !important;
        color: white !important;
    }
    .stSelectbox > div > div:hover {
        border-color: #52b7ff !important;
    }
    /* Links */
    a {
        color: #52b7ff;
        text-decoration: none;
        transition: all 0.2s ease;
    }
    a:hover {
        color: #1e90ff;
        text-decoration: none;
    }
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(30, 32, 47, 0.95);
    }
    ::-webkit-scrollbar-thumb {
        background: #52b7ff;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #1e90ff;
    }
    /* Sidebar text colors */
    .sidebar .sidebar-content {
        color: white;
    }
    .sidebar h4 {
        color: white;
    }
    .sidebar ul {
        color: #a0aec0;
    }
    </style>
""", unsafe_allow_html=True)

# Instantiate the transformer summarization pipeline globally
transformer_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ------------------- Helper Functions -------------------
def safe_translate(text, target_language, chunk_size=4900, max_retries=3):
    """
    Translate text in chunks to avoid deep_translator length limits.
    Retries translation up to max_retries times. On failure, returns the original chunk.
    """
    translator = GoogleTranslator(source='auto', target=target_language)
    translated_text = ""
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        for attempt in range(max_retries):
            try:
                translated_text += translator.translate(chunk)
                break  # Break out of retry loop on success
            except RequestError as e:
                if attempt == max_retries - 1:
                    translated_text += chunk  # Fallback: append original text
                else:
                    time.sleep(1)  # Wait before retrying
    return translated_text

def transformer_summarize(text: str, summarizer, max_chunk_size: int = 1000, max_length: int = 130, min_length: int = 30) -> str:
    """
    Summarize a long text using a transformer summarization pipeline.
    The text is split into chunks (based on sentence boundaries) to avoid token length issues.
    """
    if not text:
        return ""
    nltk.download('punkt', quiet=True)
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    summary_text = ""
    for chunk in chunks:
        try:
            summarized = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summary_text += summarized[0]['summary_text'] + " "
        except Exception as e:
            st.error(f"Error during transformer summarization: {str(e)}")
            summary_text += chunk + " "
    return summary_text.strip()

# ------------------- NewsSearcher -------------------
class NewsSearcher:
    def __init__(self):
        self.config = Config()
        self.config.browser_user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
        self.search_settings = {
            'region': 'in-en',
            'safesearch': 'off',
            'timelimit': 'm',
            'max_results': 3
        }

    def search_news(self, query: str, location: str = None) -> List[Dict]:
        articles = []
        try:
            keywords = f"{query} {location} news -site:msn.com -site:usnews.com" if location else f"{query} news -site:msn.com -site:usnews.com"
            keywords = keywords.strip().replace("  ", " ")
            with DDGS() as ddgs:
                results = list(ddgs.news(
                    keywords=keywords,
                    region=self.search_settings['region'],
                    safesearch=self.search_settings['safesearch'],
                    timelimit=self.search_settings['timelimit'],
                    max_results=self.search_settings['max_results']
                ))
                for result in results:
                    article = {
                        'url': result['url'],
                        'source': result['source'],
                        'title': result['title'],
                        'text': result['body'],
                        'publish_date': result['date'],
                        'image_url': result.get('image', None)
                    }
                    articles.append(article)
        except Exception as e:
            st.error(f"Error in DuckDuckGo news search: {str(e)}")
        return articles

# ------------------- NewsProcessor -------------------
class NewsProcessor:
    def __init__(self):
        try:
            nltk.download(['punkt', 'stopwords', 'averaged_perceptron_tagger'], quiet=True)
            self.stopwords = set(stopwords.words('english') + list(punctuation))
        except Exception:
            self.stopwords = set(list(punctuation))

    def fetch_article(self, url: str) -> dict:
        try:
            config = Config()
            config.browser_user_agent = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
            article = Article(url, config=config)
            article.download()
            time.sleep(1)
            article.parse()
            text = article.text.replace('\n', ' ').replace('\r', '')
            return {
                'title': article.title,
                'text': text,
                'url': url,
                'publish_date': article.publish_date,
                'image_url': article.top_image
            }
        except Exception:
            return {
                'title': "Article Preview Unavailable",
                'text': "Full article content could not be retrieved. You can visit the original source for complete information.",
                'url': url,
                'publish_date': None,
                'image_url': None
            }

    def summarize_text(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """
        Summarizes the provided text using the transformer summarizer.
        """
        if not text:
            return ""
        try:
            return transformer_summarize(text, transformer_summarizer, max_chunk_size=1000, max_length=max_length, min_length=min_length)
        except Exception as e:
            st.error(f"Error in summarization: {str(e)}")
            return text[:500] + "..."

# ------------------- HashnodePublisher -------------------
class HashnodePublisher:
    def __init__(self):
        self.api_token = "7d406b94-4b5b-4d53-8814-5a6a957a9564"
        self.publication_id = "67bb4bc06a1a10a27a4c1c07"
        self.api_url = "https://gql.hashnode.com/"
        self.headers = {
            'Authorization': self.api_token,
            'Content-Type': 'application/json'
        }
        try:
            nltk.download(['punkt', 'stopwords'], quiet=True)
        except:
            pass

    def _create_post_mutation(self) -> str:
        return """
        mutation PublishPost($input: PublishPostInput!) {
            publishPost(input: $input) {
                post {
                    id
                    title
                    slug
                    url
                }
            }
        }
        """

    def _slugify(self, text: str) -> str:
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        text = text.lower().strip()
        slug = re.sub(r'[^a-z0-9]+', '-', text)
        slug = slug.strip('-')
        return slug[:250]

    def _summarize_text(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """
        Uses the transformer summarizer to summarize combined article text.
        """
        if not text:
            return ""
        try:
            return transformer_summarize(text, transformer_summarizer, max_chunk_size=1000, max_length=max_length, min_length=min_length)
        except Exception as e:
            st.error(f"Error in summarization: {str(e)}")
            return text[:500] + "..."

    def generate_image(self, article: dict) -> str:
        try:
            prompt = article.get('title', '')
            summary = article.get('summary', '')
            if summary:
                prompt += f" - {summary[:100]}"
            encoded_prompt = quote(prompt, safe='')
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            response = requests.head(image_url)
            if response.status_code == 200:
                return image_url
            else:
                return None
        except Exception as e:
            return None

    def publish_combined_article(self, articles, topic: str, location: str = None, language: str = "en") -> dict:
        for article in articles:
            ai_image = self.generate_image(article)
            if ai_image:
                article['ai_image_url'] = ai_image

        original_title = f"News Roundup: {topic.title()}"
        if location:
            original_title += f" in {location.title()}"
        slug = self._slugify(original_title)
        if not slug:
            slug = f"news-roundup-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        if language != "en":
            display_title = safe_translate(original_title, language)
        else:
            display_title = original_title

        content = self.format_combined_content(articles, topic, location, language)
        
        cover_image = None
        if articles and articles[0].get('image_url'):
            cover_image_url = articles[0]['image_url'].rstrip("\\/")
            cover_image = {"coverImageURL": cover_image_url}
        
        variables = {
            "input": {
                "title": display_title,
                "contentMarkdown": content,
                "slug": slug,
                "publicationId": self.publication_id,
                "tags": [
                    {"name": "News", "slug": "news"}
                ],
                "disableComments": False,
                "coverImageOptions": cover_image
            }
        }
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={'query': self._create_post_mutation(), 'variables': variables}
            )
            if response.status_code == 200:
                result = response.json()
                if 'errors' in result:
                    st.error(f"Hashnode API Error:\n{json.dumps(result['errors'], indent=2)}")
                    return None
                return result.get('data', {}).get('publishPost', {}).get('post')
            else:
                st.error(f"HTTP Error: {response.status_code}\nResponse: {response.text}")
                return None
        except Exception as e:
            st.error(f"Error publishing article: {str(e)}")
            return None

    def format_combined_content(self, articles, topic: str, location: str = None, language: str = "en") -> str:
        current_date = datetime.now().strftime("%Y-%m-%d")
        combined_text = ""
        for article in articles:
            if article.get('text'):
                combined_text += article['text'] + " "
            elif article.get('summary'):
                combined_text += article['summary'] + " "
        combined_summary = self._summarize_text(combined_text, max_length=130, min_length=30)
        content = f"# News Roundup: {topic.title()}"
        if location:
            content += f" in {location.title()}"
        content += f"\n\n*Published on {current_date}*\n\n"
        content += "## Introduction\n"
        content += f"Below you'll find a curated overview of the latest news about **{topic}**"
        if location:
            content += f" in **{location}**"
        content += ". This post aggregates multiple sources and includes both original and AI-generated images.\n\n"
        content += "## Combined Summary\n"
        content += combined_summary + "\n\n"
        content += "## Detailed Summaries\n\n"
        for idx, article in enumerate(articles, 1):
            title = article.get('title', '').strip() or f"Article #{idx}"
            content += f"### {idx}. {title}\n\n"
            source_name = article.get('source', 'Unknown Source')
            source_url = article.get('url', '')
            content += f"**Source**: {source_name}\n\n"
            if source_url:
                content += f"**Read Full Article**: [Link]({source_url})\n\n"
            per_article_summary = article.get('summary', '')
            if per_article_summary:
                content += f"**Article Summary**:\n\n{per_article_summary}\n\n"
            if article.get('image_url'):
                content += "**Original Image**:\n\n"
                content += f"![Original Article Image]({article['image_url']})\n\n"
            if article.get('ai_image_url'):
                content += "**AI-Generated Illustration**:\n\n"
                content += f"![AI Generated Illustration]({article['ai_image_url']})\n\n"
                content += "*AI-generated image related to this article.*\n\n"
            content += "---\n\n"
        content += "\n\n---\n"
        content += "*This news roundup was automatically curated and published using AI. "
        content += f"Last updated: {current_date}*"
        if language != "en":
            content = safe_translate(content, language)
        return content

# ------------------- Streamlit App -------------------
def main():
    # Header
    st.markdown("""
        <div class="header-container">
            <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">📰 AI News Hub</h1>
            <p style="font-size: 1.1rem; opacity: 0.9;">Discover, Summarize, and Share News with AI</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.header("⚙️ Settings")
        
        language = st.selectbox(
            "Select Language",
            {
                "en": "English 🇬🇧",
                "es": "Spanish 🇪🇸",
                "fr": "French 🇫🇷",
                "de": "German 🇩🇪",
                "it": "Italian 🇮🇹",
                "pt": "Portuguese 🇵🇹",
                "ru": "Russian 🇷🇺",
                "zh": "Chinese 🇨🇳",
                "ja": "Japanese 🇯🇵",
                "ko": "Korean 🇰🇷"
            }
        )
        
        location = st.text_input("📍 Location (Optional)", placeholder="e.g., New York, London, Tokyo")
        
        st.markdown("""
            <div style="margin-top: 2rem;">
                <h4>Powered by:</h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li>🤖 BART AI Summarization</li>
                    <li>🌐 DuckDuckGo News Search</li>
                    <li>🔄 Google Translate</li>
                    <li>📝 Hashnode Publishing</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        topic = st.text_input("🔍 Enter your news topic", placeholder="e.g., AI, Climate Change, Technology")
        if st.button("Search News", use_container_width=True):
            if topic:
                with st.spinner("🔍 Searching for news articles..."):
                    searcher = NewsSearcher()
                    articles = searcher.search_news(topic, location)
                    
                    if articles:
                        st.success(f"Found {len(articles)} relevant articles!")
                        for article in articles:
                            with st.container():
                                st.markdown(f"""
                                    <div class="article-card">
                                        <div class="article-title">{article['title']}</div>
                                        <div class="article-summary">{article['text'][:300]}...</div>
                                        <div class="article-meta">
                                            <span class="source-tag">{article['source']}</span>
                                            <span>📅 {article['publish_date']}</span>
                                            <a href="{article['url']}" target="_blank">🔗 Read More</a>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("No articles found. Try different search terms or location.")
            else:
                st.warning("Please enter a topic to search.")

    with col2:
        st.markdown('<div style="background-color: white; padding: 1.5rem; border-radius: 12px;">', unsafe_allow_html=True)
        st.header("📝 Article Actions")
        
        if 'articles' in locals():
            if st.button("✨ Generate Summary", use_container_width=True):
                with st.spinner("🤖 Generating AI summary..."):
                    processor = NewsProcessor()
                    summaries = []
                    for article in articles:
                        summary = processor.summarize_text(article['text'])
                        summaries.append(summary)
                    
                    st.markdown("### 📋 Combined Summary")
                    for i, summary in enumerate(summaries, 1):
                        st.markdown(f"""
                            <div class="article-card">
                                <div class="article-title">Article {i}</div>
                                <div class="article-summary">{summary}</div>
                            </div>
                        """, unsafe_allow_html=True)

            if st.button("🚀 Publish to Hashnode", use_container_width=True):
                with st.spinner("📤 Publishing to Hashnode..."):
                    publisher = HashnodePublisher()
                    result = publisher.publish_combined_article(articles, topic, location, language)
                    if result and result.get('post'):
                        st.success("✅ Article published successfully!")
                        st.markdown(f"""
                            <div class="article-card">
                                <div class="article-title">Published on Hashnode</div>
                                <div class="article-summary">
                                    Your article has been published! Click below to view it.
                                </div>
                                <a href="{result['post']['url']}" target="_blank" style="
                                    display: inline-block;
                                    margin-top: 1rem;
                                    padding: 0.5rem 1rem;
                                    background-color: #0066cc;
                                    color: white;
                                    text-decoration: none;
                                    border-radius: 6px;
                                    font-weight: 500;
                                ">View on Hashnode</a>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"Failed to publish: {result.get('error', 'Unknown error')}")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()