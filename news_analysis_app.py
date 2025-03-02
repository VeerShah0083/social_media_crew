import getpass
import os
import warnings
import time
from openai import OpenAI
import yaml
from typing import List, Dict, Any, Optional
import threading
import streamlit as st
from streamlit_echarts import st_echarts
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
#import numpy as np
import io
import base64
from PIL import Image
from pydantic import BaseModel, Field, ConfigDict, field_validator
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
from dotenv import load_dotenv

# Suppress Pydantic deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

def setup_api_keys(force_reset=False):
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or force_reset:
        openai_key = st.text_input("Enter your OpenAI API key:", type="password")
        os.environ["OPENAI_API_KEY"] = openai_key

    serper_key = os.getenv("SERPER_API_KEY")
    if not serper_key or force_reset:
        serper_key = st.text_input("Enter your Serper API key:", type="password")
        os.environ["SERPER_API_KEY"] = serper_key

    os.environ['OPENAI_MODEL_NAME'] = 'gpt-4o-mini'
    
    return bool(openai_key and serper_key)

class SourceReliability(BaseModel):
    domain: str
    factual_rating: str  # High, Low, Mixed, Mostly Factual
    articles_count: int
    engagement: int

class SocialMediaMetrics(BaseModel):
    hashtag: str
    engagement_rate: float  # Percentage
    reach: int
    sentiment: str  # Positive, Negative, Neutral

class ContentAnalysisMetrics(BaseModel):
    language_percentage: float
    coordination_percentage: float
    source_percentage: float
    bot_like_activity_percentage: float

class TimeSeriesData(BaseModel):
    date: str
    count: int

class PropagandaTechnique(BaseModel):
    technique_name: str  # e.g., "Appeal to fear", "False equivalence", "Strawman"
    frequency: int  # How many instances detected
    severity: float  # 0-10 scale of severity
    example: str  # A brief example from the article
    explanation: str  # Why this is considered propaganda

class MisinformationIndicator(BaseModel):
    indicator_type: str  # e.g., "Factual error", "Missing context", "Manipulated content"
    confidence: float  # 0-1 scale of confidence in detection
    correction: str  # The factual correction or missing context
    source_verification: List[str]  # Sources that verify/contradict

class CoordinationPattern(BaseModel):
    pattern_type: str  # e.g., "Identical phrasing", "Synchronized publishing", "Cross-platform amplification"
    strength: float  # 0-1 scale of coordination strength
    entities_involved: List[str]  # Websites, accounts, networks involved
    timeline: str  # Brief description of coordination timeline

class BotActivityMetrics(BaseModel):
    bot_likelihood_score: float  # 0-1 scale 
    account_creation_patterns: str  # Description of suspicious patterns
    behavioral_indicators: List[str]  # List of indicators suggesting bot activity
    network_analysis: str  # Brief description of network behavior

class FakeNewsSite(BaseModel):
    domain: str
    shares: int
    engagement: int
    known_false_stories: int
    verification_failures: List[str]  # List of fact-checking failures
    deceptive_practices: List[str]  # Deceptive practices employed
    network_connections: List[str]  # Connected entities in disinformation network

class EnhancedPropagandaAnalysis(BaseModel):
    overall_reliability_score: float  # 0-100 scale
    propaganda_techniques: List[PropagandaTechnique]
    misinformation_indicators: List[MisinformationIndicator]
    coordination_patterns: List[CoordinationPattern]
    bot_activity_metrics: BotActivityMetrics
    fake_news_sites: List[FakeNewsSite]
    manipulation_timeline: List[Dict[str, Any]]  # Timeline of information manipulation
    narrative_fingerprint: Dict[str, float]  # Distinct narrative patterns and their strength
    cross_verification_results: Dict[str, Any]  # Results of cross-verification with reliable sources
    recommended_verification_steps: List[str]  # Recommended steps for readers to verify content


class NewsAnalysisReport(BaseModel):
    query_summary: str
    key_findings: str
    related_articles: List[Dict[str, str]]  # {title: str, url: str}
    related_words: List[str]  # For wordcloud
    topic_clusters: List[Dict[str, Any]]  # {topic: str, size: int, related_narratives: List[str]}
    top_sources: List[SourceReliability]
    top_hashtags: List[SocialMediaMetrics]
    similar_posts_time_series: List[TimeSeriesData]
    fake_news_sites: List[Dict[str, Any]]  # {site: str, shares: int}
    content_analysis: ContentAnalysisMetrics
    propaganda_analysis: EnhancedPropagandaAnalysis
    platform_facts: List[str]
    cross_source_facts: List[str]

    model_config = ConfigDict(arbitrary_types_allowed=True)

def create_news_analysis_agents():
    return [
        Agent(
            role="Web Crawler",
            goal="Extract news data for the query",
            backstory="An expert web crawler specialized in news sites, capable of identifying reliable sources and extracting relevant articles efficiently.",
            tools=[ScrapeWebsiteTool(), WebsiteSearchTool(), SerperDevTool()],
            tool_calls=True,
            verbose=True
        ),
        Agent(
            role="News Content Analyst",
            goal="Analyze news content in depth",
            backstory="A seasoned journalist with expertise in fact-checking, source reliability assessment, and content analysis who can identify credible sources, biases, and trends in news articles.",
            tools=[ScrapeWebsiteTool(), WebsiteSearchTool(), SerperDevTool()],
            tool_calls=True,
            verbose=True
        ),
        Agent(
            role="Social Media Tracking Specialist",
            goal="Track news spread on social media",
            backstory="A social media expert who specializes in tracking how news spreads across platforms, identifying trending hashtags, measuring engagement, and analyzing sentiment related to news topics.",
            tools=[WebsiteSearchTool(), SerperDevTool()],
            tool_calls=True,
            verbose=True
        ),
        Agent(
            role="News Data Visualization Expert",
            goal="Create data visualizations from news analysis",
            backstory="A data visualization specialist who transforms news analysis data into meaningful visual representations including topic clusters, wordclouds, time series graphs, and reliability charts.",
            tools=[SerperDevTool()],
            tool_calls=True,
            verbose=True
        ),
        Agent(
        role="Propaganda & Misinformation Analyst",
        goal="Identify and quantify propaganda, misinformation, and coordinated inauthentic behavior in news content",
        backstory="""An expert with advanced training in computational propaganda detection, 
                   misinformation analysis, and network forensics. Specialized in identifying 
                   manipulation techniques, assessing credibility signals, detecting narrative 
                   manipulation, and tracing the spread of false information across media ecosystems.
                   Has experience working with fact-checking organizations and research institutions 
                   on digital media literacy.""",
        tools=[ScrapeWebsiteTool(), SerperDevTool(), WebsiteSearchTool()],
        tool_calls=True,
        verbose=True,
        allow_delegation=True
        ),
        Agent(
            role="News Report Generator",
            goal="Compile findings into a comprehensive news analysis report",
            backstory="A professional report writer specialized in organizing complex news analysis data into structured, insightful, and actionable reports with clear visualizations and fact comparisons.",
            tool_calls=False,
            verbose=True
        )
    ]

def create_news_analysis_tasks(agents, user_query, urls=None, hashtags=None, keywords=None):
    return [
        Task(
            description=f"Crawl news websites for articles related to: {user_query}. Identify reliable and unreliable sources. Extract article URLs, publication dates, and engagement metrics.",
            agent=agents[0],
            expected_output="A comprehensive dataset of news articles with their sources, reliability metrics, and engagement statistics."
        ),
        Task(
            description=f"Analyze the content of collected news articles for: {user_query}. Extract key findings, related topics, narrative patterns, and assess the factual nature of the content.",
            agent=agents[1],
            expected_output="Content analysis including key findings, related words for wordcloud, topic clusters, and fact assessments from multiple sources."
        ),
        Task(
            description=f"Track how the news topic '{user_query}' is spreading on social media. Identify top hashtags, engagement rates, reach, sentiment, and track similar posts over time.",
            agent=agents[2],
            expected_output="Social media analysis report with top hashtags, engagement metrics, sentiment analysis, and temporal spread patterns."
        ),
        Task(
            description="Generate data visualization structures for topic clusters, wordclouds, time series of news spread, and source reliability comparisons.",
            agent=agents[3],
            expected_output="Data structures ready for visualization including topic clusters with size metrics, temporal data for time series, and comparative source reliability data."
        ),
        Task(
            description=f"""Conduct a comprehensive analysis of news content related to '{user_query}' 
            for propaganda, misinformation, and coordinated inauthentic behavior.
        
            1. IDENTIFY PROPAGANDA TECHNIQUES:
              - Detect specific propaganda techniques (name-calling, bandwagon, testimonial, etc.)
              - Rate severity and provide concrete examples from articles
              - Calculate frequency of each technique across sources
        
            2. ASSESS MISINFORMATION INDICATORS:
              - Fact-check key claims against verified information
              - Identify missing context that changes interpretation
              - Document factual errors with correction sources
              - Evaluate manipulated quotes, images, or statistics
        
            3. DETECT COORDINATION PATTERNS:
              - Identify synchronized publishing or messaging
              - Track identical phrasing across seemingly unrelated sources
              - Analyze cross-platform narrative amplification
              - Map connections between sources spreading similar misinformation
        
            4. MEASURE BOT-LIKE ACTIVITY:
              - Calculate bot likelihood scores for sharing patterns
              - Identify suspicious account behaviors and creation patterns
              - Analyze network spread characteristics typical of inauthentic amplification
        
            5. CATALOG FAKE NEWS SITES:
              - Identify highest-impact fake news domains by engagement metrics
              - Document history of verification failures
              - Detail deceptive practices employed
              - Map network connections to other disinformation sources
        
            6. DEVELOP VERIFICATION GUIDANCE:
              - Create step-by-step verification process for readers
              - Suggest credible alternative sources for verification
              - Provide red flags that indicate potential misinformation
        
            Use tools to scrape articles, analyze text patterns, and verify claims against reliable 
            sources. Quantify results where possible with specific metrics and confidence scores.
            """,
            agent=agents[4],
            expected_output="""Comprehensive propaganda and misinformation analysis with:
            1. Overall reliability score with confidence intervals
            2. Cataloged propaganda techniques with examples and frequency metrics
            3. Fact-check results with verification sources
            4. Coordination pattern analysis with network visualization data
            5. Bot activity metrics with detailed behavioral indicators
            6. Ranked list of fake news sites with engagement metrics and verification history
            7. Timeline showing evolution of misinformation spread
            8. Narrative fingerprint showing distinctive patterns across sources
            9. Reader guidance for information verification""",
        ),
        Task(
            description="Generate final comprehensive news analysis report integrating all findings.",
            agent=agents[5],
            expected_output="A structured news analysis report summarizing all findings with clear sections for key insights, source reliability, content analysis, and fact comparisons.",
            output_pydantic=NewsAnalysisReport
        )
    ]

def create_news_analysis_crew(user_query, urls=None, hashtags=None, keywords=None):
    agents = create_news_analysis_agents()
    tasks = create_news_analysis_tasks(agents, user_query, urls, hashtags, keywords)
    return Crew(agents=agents, tasks=tasks, process="sequential", verbose=True)

def run_news_analysis(user_query, urls=None, hashtags=None, keywords=None, progress_callback=None):
    crew = create_news_analysis_crew(user_query, urls, hashtags, keywords)
    
    # If a progress callback was provided, update it with each step
    if progress_callback:
        # Get the total number of tasks
        total_tasks = len(crew.tasks)
        current_task = 0
        
        # Create a wrapper for the progress callback
        def progress_wrapper(message):
            nonlocal current_task
            if "starting" in message.lower():
                current_task += 1
            progress = current_task / total_tasks
            progress_callback(progress, message)
        
        # Set the crew's verbose_callback to our wrapper
        crew.verbose_callback = progress_wrapper
    
    result = crew.kickoff(inputs={'query': user_query, 'urls': urls, 'hashtags': hashtags, 'keywords': keywords})
    return result.pydantic

# Function to generate a word cloud image
def generate_wordcloud(words):
    # If we just have a list of words, convert to frequency dict
    if isinstance(words, list):
        word_freq = {word: words.count(word) for word in set(words)}
    else:
        word_freq = words
    
    wc = WordCloud(width=800, height=400, background_color='white')
    wc.generate_from_frequencies(word_freq)
    
    # Convert to image
    img = wc.to_image()
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Streamlit App
def streamlit_app():
    st.set_page_config(page_title="Real-Time News Analysis Dashboard", layout="wide")
    
    st.title("Real-Time News Analysis Dashboard")
    st.markdown("This dashboard provides real-time analysis of news articles and social media trends on your chosen topic.")
    
    # Setup container for API keys
    with st.expander("API Key Setup"):
        api_keys_ready = setup_api_keys()
    
    # Query input
    user_query = st.text_input("Enter news topic to analyze:", placeholder="e.g., Climate Change, Cryptocurrency, etc.")
    
    # Optional inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        include_urls = st.checkbox("Include specific news URLs?")
        urls = st.text_input("Enter news URLs (comma-separated):", disabled=not include_urls)
    
    with col2:
        track_hashtags = st.checkbox("Track specific hashtags?")
        hashtags = st.text_input("Enter hashtags to track (comma-separated):", disabled=not track_hashtags)
    
    with col3:
        include_keywords = st.checkbox("Include additional keywords?")
        keywords = st.text_input("Enter additional keywords (comma-separated):", disabled=not include_keywords)
    
    # Process URLs, hashtags, and keywords
    processed_urls = urls.split(',') if include_urls and urls else None
    processed_hashtags = hashtags.split(',') if track_hashtags and hashtags else None
    processed_keywords = keywords.split(',') if include_keywords and keywords else None
    
    # State variables for tracking analysis progress
    if 'analysis_running' not in st.session_state:
        st.session_state.analysis_running = False
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'progress' not in st.session_state:
        st.session_state.progress = 0.0
    if 'progress_message' not in st.session_state:
        st.session_state.progress_message = ""
    
    # Function to update progress
    def update_progress(progress, message):
        st.session_state.progress = progress
        st.session_state.progress_message = message
    
    # Analyze button
    analyze_btn = st.button("Analyze News", disabled=not user_query or not api_keys_ready or st.session_state.analysis_running)
    
    # Progress bar
    if st.session_state.analysis_running:
        progress_bar = st.progress(st.session_state.progress)
        st.write(st.session_state.progress_message)
    
    # Run analysis in a separate thread to avoid blocking the UI
    if analyze_btn:
        st.session_state.analysis_running = True
        st.session_state.analysis_complete = False
        st.session_state.analysis_result = None
        st.session_state.progress = 0.0
        st.session_state.progress_message = "Starting analysis..."
        
        def run_analysis_thread():
            try:
                result = run_news_analysis(
                    user_query, 
                    processed_urls, 
                    processed_hashtags, 
                    processed_keywords,
                    progress_callback=update_progress
                )
                st.session_state.analysis_result = result
                st.session_state.analysis_complete = True
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
            finally:
                st.session_state.analysis_running = False
        
        # Start analysis in a background thread
        analysis_thread = threading.Thread(target=run_analysis_thread)
        analysis_thread.start()
        
        # Force a rerun to show the progress bar
        st.experimental_rerun()
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.analysis_result:
        report = st.session_state.analysis_result
        
        # Create tabs for different sections of the report
        tabs = st.tabs([
            "Summary", 
            "Articles & Topics", 
            "Sources & Hashtags", 
            "Time Series & Content Analysis", 
            "Propaganda Analysis",
            "Fact Comparison"
        ])
        
        # Summary Tab
        with tabs[0]:
            st.header(f"News Analysis: {report.query_summary}")
            st.subheader("Key Findings")
            st.write(report.key_findings)
        
        # Articles & Topics Tab
        with tabs[1]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Related Articles")
                for article in report.related_articles:
                    for title, url in article.items():
                        st.markdown(f"- [{title}]({url})")
            
            with col2:
                st.subheader("Related Words")
                wordcloud_img = generate_wordcloud(report.related_words)
                st.image(wordcloud_img, use_column_width=True)
            
            st.subheader("Topic Clusters")
            # Create bubble chart for topic clusters
            topic_data = []
            for cluster in report.topic_clusters:
                topic = cluster.get('topic', 'Unknown')
                size = cluster.get('size', 10)
                narratives = cluster.get('related_narratives', [])
                narrative_text = ", ".join(narratives) if narratives else "No related narratives"
                topic_data.append({
                    'name': topic,
                    'value': size,
                    'narratives': narrative_text
                })
            
            # Create bubble chart options
            bubble_options = {
                'tooltip': {
                    'formatter': '{b}: {c}<br/>{@narratives}'
                },
                'series': [{
                    'type': 'graph',
                    'layout': 'force',
                    'force': {
                        'repulsion': 100,
                        'edgeLength': 30
                    },
                    'roam': True,
                    'label': {
                        'show': True
                    },
                    'data': [
                        {
                            'name': item['name'],
                            'value': item['value'],
                            'symbolSize': item['value'] * 2,  # Scale bubble size
                            'narratives': item['narratives']
                        }
                        for item in topic_data
                    ]
                }]
            }
            
            st_echarts(options=bubble_options, height=400)
        
        # Sources & Hashtags Tab
        with tabs[2]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top Sources")
                source_data = []
                for source in report.top_sources:
                    source_data.append({
                        'Domain': source.domain,
                        'Factual Rating': source.factual_rating,
                        'Articles': source.articles_count,
                        'Engagement': source.engagement
                    })
                
                source_df = pd.DataFrame(source_data)
                st.dataframe(source_df, use_container_width=True)
                
                # Create a bar chart for source reliability
                fig = px.bar(
                    source_df, 
                    x='Domain', 
                    y='Engagement',
                    color='Factual Rating',
                    title='Top Sources by Engagement',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Top Hashtags")
                hashtag_data = []
                for hashtag in report.top_hashtags:
                    hashtag_data.append({
                        'Hashtag': hashtag.hashtag,
                        'Engagement Rate (%)': hashtag.engagement_rate,
                        'Reach': hashtag.reach,
                        'Sentiment': hashtag.sentiment
                    })
                
                hashtag_df = pd.DataFrame(hashtag_data)
                st.dataframe(hashtag_df, use_container_width=True)
                
                # Create a colored bar chart for hashtags by sentiment
                fig = px.bar(
                    hashtag_df,
                    x='Hashtag',
                    y='Reach',
                    color='Sentiment',
                    title='Top Hashtags by Reach and Sentiment',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Time Series & Content Analysis Tab
        with tabs[3]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Similar Posts Spread Over Time")
                time_data = []
                for data_point in report.similar_posts_time_series:
                    time_data.append({
                        'Date': data_point.date,
                        'Posts': data_point.count
                    })
                
                time_df = pd.DataFrame(time_data)
                fig = px.line(
                    time_df,
                    x='Date',
                    y='Posts',
                    title='Post Volume Over Time',
                    markers=True,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Most Shared Fake News Sites")
                fake_news_data = []
                for site in report.fake_news_sites:
                    site_name = site.get('site', 'Unknown')
                    shares = site.get('shares', 0)
                    fake_news_data.append({
                        'Site': site_name,
                        'Shares': shares
                    })
                
                fake_news_df = pd.DataFrame(fake_news_data)
                fig = px.bar(
                    fake_news_df,
                    x='Site',
                    y='Shares',
                    title='Most Shared Fake News Sites',
                    color='Shares',
                    color_continuous_scale='Reds',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Content Analysis Metrics")
            metrics = report.content_analysis
            metrics_data = {
                'Metric': ['Language', 'Coordination', 'Source', 'Bot-like Activity'],
                'Percentage': [
                    metrics.language_percentage,
                    metrics.coordination_percentage,
                    metrics.source_percentage,
                    metrics.bot_like_activity_percentage
                ]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            fig = px.bar(
                metrics_df,
                x='Metric',
                y='Percentage',
                title='Content Analysis Metrics',
                color='Percentage',
                color_continuous_scale='Blues',
                height=400
            )
            fig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        
        # Propaganda Analysis Tab
        with tabs[4]:
            propaganda = report.propaganda_analysis
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.metric(
                    "Overall Reliability Score", 
                    f"{propaganda.overall_reliability_score}/100",
                    delta=f"{propaganda.overall_reliability_score - 50:.1f}" if propaganda.overall_reliability_score != 50 else None
                )
            
            with col2:
                # Create gauge chart for reliability score
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=propaganda.overall_reliability_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Reliability Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 33], 'color': "red"},
                            {'range': [33, 66], 'color': "yellow"},
                            {'range': [66, 100], 'color': "green"}
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Propaganda Techniques Detected")
            technique_data = []
            for technique in propaganda.propaganda_techniques:
                technique_data.append({
                    'Technique': technique.technique_name,
                    'Frequency': technique.frequency,
                    'Severity': technique.severity,
                    'Example': technique.example,
                    'Explanation': technique.explanation
                })
            
            technique_df = pd.DataFrame(technique_data)
            if not technique_df.empty:
                st.dataframe(technique_df, use_container_width=True)
                
                # Create heatmap for propaganda techniques
                fig = px.scatter(
                    technique_df,
                    x='Frequency',
                    y='Technique',
                    size='Severity',
                    color='Severity',
                    color_continuous_scale='Reds',
                    title='Propaganda Techniques by Frequency and Severity',
                    height=400,
                    hover_data=['Example']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No propaganda techniques detected")
            
            st.subheader("Misinformation Indicators")
            misinfo_data = []
            for indicator in propaganda.misinformation_indicators:
                misinfo_data.append({
                    'Type': indicator.indicator_type,
                    'Confidence (%)': indicator.confidence * 100,
                    'Correction': indicator.correction,
                    'Sources': ', '.join(indicator.source_verification)
                })
            
            misinfo_df = pd.DataFrame(misinfo_data)
            if not misinfo_df.empty:
                st.dataframe(misinfo_df, use_container_width=True)
            else:
                st.info("No misinformation indicators detected")
            
            st.subheader("Bot Activity Metrics")
            bot = propaganda.bot_activity_metrics
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Bot Likelihood Score", f"{bot.bot_likelihood_score * 100:.1f}%")
                st.write(f"**Account Creation Patterns:** {bot.account_creation_patterns}")
                
                st.write("**Behavioral Indicators:**")
                for indicator in bot.behavioral_indicators:
                    st.write(f"- {indicator}")
            
            with col2:
                # Create gauge for bot likelihood
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=bot.bot_likelihood_score * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Bot Likelihood Score"},
                    number={'suffix': '%'},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 33], 'color': "green"},
                            {'range': [33, 66], 'color': "yellow"},
                            {'range': [66, 100], 'color': "red"}
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                st.write(f"**Network Analysis:** {bot.network_analysis}")
            
            st.subheader("Information Manipulation Timeline")
            timeline_data = []
            for entry in propaganda.manipulation_timeline:
                timeline_data.append({
                    'Date': entry.get('date', 'Unknown'),
                    'Event': entry.get('event', 'Unknown event')
                })
            
            timeline_df = pd.DataFrame(timeline_data)
            if not timeline_df.empty:
                st.dataframe(timeline_df, use_container_width=True)
            else:
                st.info("No manipulation timeline available")
            
            st.subheader("How to Verify This Information")
            for i, step in enumerate(propaganda.recommended_verification_steps, 1):
                st.write(f"{i}. {step}")
        
        # Fact Comparison Tab
        with tabs[5]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Facts Gathered from Platform")
                for fact in report.platform_facts:
                    st.write(f"- {fact}")
            with col2:
                st.subheader("Cross-Source Facts")
                for fact in report.cross_source_facts:
                    st.write(f"- {fact}")
            
            # Add a visualization comparing cross-platform fact consistency
            st.subheader("Fact Verification Analysis")
            
            # Create a simple comparison visualization
            cross_verification = propaganda.cross_verification_results
            if cross_verification:
                verification_data = []
                for source, data in cross_verification.items():
                    if isinstance(data, dict):
                        verification_data.append({
                            'Source': source,
                            'Agreement': data.get('agreement_percentage', 0),
                            'Contradiction': data.get('contradiction_percentage', 0),
                            'Unverifiable': 100 - (data.get('agreement_percentage', 0) + data.get('contradiction_percentage', 0))
                        })
                
                if verification_data:
                    verification_df = pd.DataFrame(verification_data)
                    fig = px.bar(
                        verification_df,
                        x='Source',
                        y=['Agreement', 'Contradiction', 'Unverifiable'],
                        title='Cross-Source Fact Verification',
                        height=400,
                        barmode='stack'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No cross-verification data available")
            else:
                st.info("No cross-verification data available")
            
            # Add a narrative fingerprint visualization
            st.subheader("Narrative Fingerprint")
            fingerprint = propaganda.narrative_fingerprint
            if fingerprint:
                fingerprint_data = []
                for narrative, strength in fingerprint.items():
                    fingerprint_data.append({
                        'Narrative': narrative,
                        'Strength': strength
                    })
                
                fingerprint_df = pd.DataFrame(fingerprint_data)
                fig = px.bar(
                    fingerprint_df,
                    x='Narrative',
                    y='Strength',
                    title='Narrative Fingerprint Analysis',
                    color='Strength',
                    color_continuous_scale='Viridis',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No narrative fingerprint data available")

#Add custom menu to download results
def add_download_options(report):
    with st.expander("Download Analysis Results"):
        # Generate report as YAML
        report_yaml = yaml.dump(report.model_dump(), default_flow_style=False)
        
        # Generate report as JSON
        report_json = report.model_dump_json(indent=2)
        
        # Generate report as CSV for metrics
        metrics_data = {
            'Metric': ['Language', 'Coordination', 'Source', 'Bot-like Activity'],
            'Percentage': [
                report.content_analysis.language_percentage,
                report.content_analysis.coordination_percentage,
                report.content_analysis.source_percentage,
                report.content_analysis.bot_like_activity_percentage
            ]
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_csv = metrics_df.to_csv(index=False)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="Download as YAML",
                data=report_yaml,
                file_name="news_analysis_report.yaml",
                mime="text/yaml"
            )
        
        with col2:
            st.download_button(
                label="Download as JSON",
                data=report_json,
                file_name="news_analysis_report.json",
                mime="application/json"
            )
        
        with col3:
            st.download_button(
                label="Download Metrics as CSV",
                data=metrics_csv,
                file_name="news_analysis_metrics.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    streamlit_app()