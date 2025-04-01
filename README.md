![image](https://github.com/user-attachments/assets/dd11bb5a-4ec1-4447-a297-4db44515e382)
---

## 🚀 QuickNews - Autonomous AI News Agent 🤖📰

**QuickNews** is an intelligent AI agent that autonomously searches, summarizes, and publishes news articles. Built as an industry-ready application, it streamlines the process of gathering and publishing news content with zero human intervention.

![frontPage](https://github.com/user-attachments/assets/68e5b3e3-c48c-473b-9715-22dd06c5465d)



### Published news site excellent SEO test proof

![image alt](https://github.com/amansaroj9616/AI-news-Summarized/blob/406e5066d144759e8c9eea3d27e2a575454631ca/seo.jpg)

### Proof

[news regrading pdf](https://github.com/amansaroj9616/AI-news-Summarized/blob/f7887cfff32f084f77f6a1e41300627d61cdfdca/published_news.pdf)

### 🎯 Project Overview
- Web crawling and data extraction from reliable news sources
- Intelligent content summarization and structuring
- SEO optimization for better discoverability
- Automated publishing to Hashnode
- Support for multiple languages
- AI-generated imagery for enhanced visual appeal

### 🛠️ Technical Approach
**Data Collection & Processing**
- Uses DuckDuckGo API for fetching recent news
- Implements `newspaper3k` for article parsing
- Utilizes NLTK for NLP and summarization


### Simple Agent (WebSearch)
- In my opinion, the most basic Agent should at least be able to choose between one or two Tools and re-elaborate the output of the action to give the user a proper and concise answer. 


**Content Generation**
- Employs extractive summarization techniques
- Generates SEO-optimized titles and descriptions
- Creates AI-generated illustrations with Pollinations.ai

**Publishing Pipeline**
- Formats content in Markdown
- Optimizes images
- Publishes to Hashnode via GraphQL API
- Supports 10+ languages using Google Translate

### ✨ Features
- 🌐 **Multi-source News Aggregation**
- 📝 **Smart Summarization**
- 🎨 **AI Image Generation**
- 🌍 **Multilingual Support**
- 📊 **Clean UI** with Streamlit
- 🚀 **One-Click Publishing** to Hashnode

### 🔧 Setup & Installation
```bash
git clone https://github.com/yourusername/quicknews.git
cd quicknews
pip install -r requirements.txt
```

Configure API tokens in `Final_name.py`:
```python
self.api_token = "Your_hashnode_api"
self.publication_id = "Publication_id"
```

Run the application:
```bash
streamlit run main.py
```

### 🎮 Usage
1. Launch the app
2. Enter your search topic
3. (Optional) Add location for specific news
4. Select display language
5. Click **Search News**
6. Review summaries
7. Click **Publish to Hashnode**

### 🔑 Required API Keys
- Hashnode API Token
- Hashnode Publication ID

### 🤝 Contributing
Feel free to fork the repo and submit pull requests. Open issues for major changes.

### 🎓 Acknowledgments
- Built for **Flipr Hackathon 25**
- Uses **newspaper3k**
- Powered by **Streamlit**
- Thanks to **Hashnode** for their GraphQL API


  📞 Contact
For any queries regarding the project, feel free to reach out:

Email: amansaroj4518@gmail.com
LinkedIn: https://www.linkedin.com/in/aman-saroj/
