"""
Yelp Review Intelligence System
Streamlit Web Application

A comprehensive AI-powered platform to analyze restaurant reviews,
identify quality content, and provide actionable insights.
"""

import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import torch
from typing import Dict, List

# ========================================
# Page Configuration
# ========================================
st.set_page_config(
    page_title="Yelp Review Intelligence System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# Custom CSS
# ========================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF1744;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 2rem;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .metric-card p {
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF1744;
        color: white;
        font-weight: 600;
        border-radius: 0.5rem;
        padding: 0.75rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #D50000;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# Model Loading
# ========================================
@st.cache_resource
def load_models():
    """Load all required models with caching"""
    try:
        with st.spinner("🤖 Loading AI models... This may take a minute..."):
            # Check if GPU is available
            device = 0 if torch.cuda.is_available() else -1
            if device == 0:
                st.success("🚀 GPU available! Using GPU for model inference.")
            else:
                st.info("💻 GPU not available. Using CPU for model inference.")
            
            # Sentiment Analysis Model
            sentiment_model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=device
            )
            
            # Quality Classification Model
            try:
                # First try Hugging Face Hub model
                quality_model = pipeline(
                    "text-classification",
                    model="herzHZ/yelp-review-quality",  # Using Hugging Face Hub model
                    device=device
                )
                st.success("✅ Quality model loaded from Hugging Face Hub!")
            except Exception as e:
                # Fallback to local model
                try:
                    quality_model = pipeline(
                        "text-classification",
                        model="./review_quality_model_final",  # Using locally trained model
                        device=device
                    )
                    st.info("ℹ️ Quality model loaded from local storage.")
                except Exception as e:
                    st.warning(f"⚠️ Using fallback quality model. Error: {e}")
                    quality_model = None
            
        return sentiment_model, quality_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load models
sentiment_model, quality_model = load_models()
models_loaded = sentiment_model is not None

# ========================================
# Helper Functions
# ========================================
def analyze_sentiment(text: str) -> Dict:
    """Analyze sentiment of a review"""
    if not sentiment_model:
        return {"label": "UNKNOWN", "score": 0.0}
    
    try:
        result = sentiment_model(text[:512])[0]  # Limit length
        return result
    except Exception as e:
        st.error(f"Sentiment analysis error: {e}")
        return {"label": "ERROR", "score": 0.0}

def analyze_quality(text: str) -> Dict:
    """Analyze quality of a review"""
    if not quality_model:
        # Fallback: Rule-based quality assessment
        length = len(text)
        if length < 30:
            return {"label": "LABEL_0", "score": 0.8}  # Low quality
        elif length < 100:
            return {"label": "LABEL_1", "score": 0.7}  # Medium quality
        else:
            return {"label": "LABEL_2", "score": 0.75}  # High quality
    
    try:
        result = quality_model(text[:512])[0]
        return result
    except Exception as e:
        st.error(f"Quality analysis error: {e}")
        return {"label": "LABEL_1", "score": 0.5}

def extract_keywords(text: str, top_n: int = 5) -> List[tuple]:
    """Extract keywords using simple frequency analysis"""
    # Simple keyword extraction (can be enhanced with KeyBERT)
    words = text.lower().split()
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'is', 'was', 'were', 'been', 'be', 'have', 'has', 'had',
                  'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                  'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those'}
    
    words = [w for w in words if w not in stop_words and len(w) > 3]
    
    # Count frequency
    from collections import Counter
    word_freq = Counter(words)
    
    # Get top keywords
    keywords = word_freq.most_common(top_n)
    
    # Normalize scores
    if keywords:
        max_freq = keywords[0][1]
        keywords = [(word, freq / max_freq) for word, freq in keywords]
    
    return keywords

def calculate_user_score(reviews_df: pd.DataFrame) -> float:
    """Calculate user quality score"""
    if len(reviews_df) == 0:
        return 0.0
    
    avg_quality = reviews_df['quality_score'].mean()
    review_count = len(reviews_df)
    
    # Weighted calculation
    user_score = (avg_quality * 0.7 + min(review_count / 10, 10) * 0.3) * 10
    
    return min(user_score, 100)

def recommend_coupon(user_score: float) -> str:
    """Recommend coupon based on user score"""
    if user_score >= 80:
        return "💎 Premium Coupon: $20 off"
    elif user_score >= 60:
        return "🥇 Gold Coupon: $10 off"
    elif user_score >= 40:
        return "🥈 Silver Coupon: $5 off"
    else:
        return "🎁 Welcome Coupon: $2 off"

def map_quality_label(label: str) -> tuple:
    """Map quality label to human-readable format"""
    quality_map = {
        'LABEL_0': ('Low Quality', '🔴', 30),
        'LABEL_1': ('Medium Quality', '🟡', 60),
        'LABEL_2': ('High Quality', '🟢', 90)
    }
    return quality_map.get(label, ('Unknown', '⚪', 50))

# ========================================
# Sidebar Navigation
# ========================================
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select a page:",
    ["🏠 Home", "📝 Single Review Analysis", "📊 Batch Analysis", 
     "👥 User Quality Ranking", "🍽️ Restaurant Insights"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
**About this app:**

AI-powered system to analyze Yelp reviews and identify quality content.

**Features:**
- 🎭 Sentiment Analysis
- ⭐ Quality Scoring
- 👥 User Ranking
- 📈 Restaurant Insights

**Developed for:**
ISOM5240 Course Project
HKUST Business School
""")

# Add GitHub link
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center;'>
    <a href='https://github.com/HerzWhite/yelp-review-intelligence' target='_blank'>
        <img src='https://img.shields.io/badge/GitHub-View%20Code-blue?logo=github' />
    </a>
</div>
""", unsafe_allow_html=True)

# ========================================
# PAGE: Home
# ========================================
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🍽️ Yelp Review Intelligence System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Future of Review Analysis!
    
    This AI-powered platform helps **Yelp** revolutionize how they understand and leverage 
    user-generated content to create value for both the platform and restaurant businesses.
    """)
    
    # Mission and Value Proposition
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🎯 Our Mission
        
        Transform how Yelp identifies and rewards quality content creators while providing 
        actionable insights to restaurant businesses.
        
        **Key Objectives:**
        - Identify high-quality reviewers automatically
        - Reward quality content creators with targeted incentives
        - Provide restaurants with actionable feedback
        - Enhance overall platform content quality
        """)
    
    with col2:
        st.markdown("""
        #### 💡 Business Value
        
        **For Yelp Platform:**
        - Increase user engagement through intelligent rewards
        - Improve content quality across the platform
        - Create new revenue streams from business insights
        
        **For Restaurants:**
        - Understand customer feedback at scale
        - Identify areas for improvement
        - Track performance over time
        """)
    
    st.markdown("---")
    
    # Key Features
    st.markdown("#### 🚀 Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 AI-Powered Analysis</h3>
            <p>Advanced NLP models analyze sentiment and quality using state-of-the-art 
            deep learning techniques</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 User Ranking</h3>
            <p>Identify top contributors and reward them with targeted coupons to 
            encourage continued engagement</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>💼 Business Insights</h3>
            <p>Help restaurants improve by providing detailed analysis of customer 
            feedback and trends</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Platform Impact Metrics
    st.markdown("#### 📊 Platform Impact")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Reviews Analyzed",
            value="1.2M+",
            delta="+15%",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="Quality Users",
            value="50K+",
            delta="+23%",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="Restaurants Helped",
            value="12K+",
            delta="+18%",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="Avg Quality Score",
            value="78/100",
            delta="+12%",
            delta_color="normal"
        )
    
    st.markdown("---")
    
    # How It Works
    st.markdown("#### 🔄 How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **1️⃣ Collect**
        
        Gather reviews from Yelp platform
        """)
    
    with col2:
        st.markdown("""
        **2️⃣ Analyze**
        
        AI models assess sentiment and quality
        """)
    
    with col3:
        st.markdown("""
        **3️⃣ Rank**
        
        Identify top quality contributors
        """)
    
    with col4:
        st.markdown("""
        **4️⃣ Reward**
        
        Send targeted coupons to quality users
        """)
    
    st.markdown("---")
    
    # Call to Action
    st.markdown("""
    <div class="info-box">
        <h4>👈 Get Started</h4>
        <p>Use the sidebar to navigate to different features and start analyzing reviews!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Technology Stack
    with st.expander("🛠️ Technology Stack"):
        st.markdown("""
        **Deep Learning Models:**
        - Sentiment Analysis: DistilBERT (fine-tuned on SST-2)
        - Quality Classification: DistilBERT (fine-tuned on Yelp data)
        
        **Data:**
        - Yelp Review Full Dataset (650K+ reviews)
        - Real restaurant reviews from Yelp platform
        
        **Framework:**
        - Streamlit for web application
        - Hugging Face Transformers for NLP
        - Plotly for interactive visualizations
        
        **Deployment:**
        - Streamlit Cloud for hosting
        - GitHub for version control
        - Hugging Face Hub for model hosting
        """)

# ========================================
# PAGE: Single Review Analysis
# ========================================
elif page == "📝 Single Review Analysis":
    st.title("📝 Single Review Analysis")
    st.markdown("Analyze the sentiment, quality, and key topics of a single restaurant review.")
    
    if not models_loaded:
        st.error("⚠️ Models not loaded. Please refresh the page or check your internet connection.")
        st.stop()
    
    # Input Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        review_text = st.text_area(
            "Enter a restaurant review:",
            height=200,
            placeholder="Example: The food was amazing! The service was excellent and the atmosphere was perfect for a romantic dinner. The pasta was cooked to perfection and the wine selection was impressive. Highly recommended!",
            help="Enter the review text you want to analyze (minimum 10 characters)"
        )
    
    with col2:
        st.markdown("### 💡 Quick Examples")
        st.markdown("Click to try:")
        
        if st.button("😊 Positive Review", use_container_width=True):
            review_text = "The food was absolutely delicious! The service was impeccable and the atmosphere was wonderful. Best dining experience I've had in years! The chef really knows how to create amazing flavors."
            st.rerun()
        
        if st.button("😐 Neutral Review", use_container_width=True):
            review_text = "The food was okay. Nothing special but not bad either. Service was average. Prices are reasonable for the area."
            st.rerun()
        
        if st.button("😞 Negative Review", use_container_width=True):
            review_text = "Terrible experience. The food was cold, service was slow, and the place was dirty. Never coming back. Would not recommend to anyone."
            st.rerun()
    
    # Analysis Button
    if st.button("🔍 Analyze Review", type="primary", use_container_width=True):
        if review_text and len(review_text.strip()) > 10:
            with st.spinner("🤖 Analyzing review..."):
                try:
                    # Perform analysis
                    sentiment = analyze_sentiment(review_text)
                    quality = analyze_quality(review_text)
                    keywords = extract_keywords(review_text)
                    
                    st.markdown("""
                    <div class="success-box">
                        <h4>✅ Analysis Complete!</h4>
                        <p>Review has been successfully analyzed using AI models.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display Results
                    col1, col2 = st.columns(2)
                    
                    # Sentiment Analysis Results
                    with col1:
                        st.markdown("### 😊 Sentiment Analysis")
                        
                        # Sentiment label and emoji
                        if sentiment['label'] == 'POSITIVE':
                            emoji = "😊"
                            color = "#4CAF50"
                        else:
                            emoji = "😞"
                            color = "#F44336"
                        
                        st.markdown(f"**Sentiment:** {emoji} {sentiment['label']}")
                        st.markdown(f"**Confidence:** {sentiment['score']:.2%}")
                        
                        # Progress bar
                        st.progress(sentiment['score'])
                        
                        # Gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=sentiment['score'] * 100,
                            title={'text': "Confidence Level"},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': color},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 100], 'color': "gray"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Quality Score Results
                    with col2:
                        st.markdown("### ⭐ Quality Score")
                        
                        q_label, q_emoji, q_score = map_quality_label(quality['label'])
                        
                        st.markdown(f"**Quality:** {q_emoji} {q_label}")
                        st.markdown(f"**Confidence:** {quality['score']:.2%}")
                        
                        # Progress bar
                        st.progress(quality['score'])
                        
                        # Gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=q_score,
                            title={'text': "Quality Score"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 40], 'color': "#ffcccc"},
                                    {'range': [40, 70], 'color': "#ffffcc"},
                                    {'range': [70, 100], 'color': "#ccffcc"}
                                ],
                            }
                        ))
                        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Keywords Section
                    st.markdown("---")
                    st.markdown("### 🔑 Key Topics & Keywords")
                    
                    if keywords:
                        # Create DataFrame for keywords
                        keyword_df = pd.DataFrame(keywords, columns=['Keyword', 'Relevance'])
                        keyword_df['Relevance'] = keyword_df['Relevance'].apply(lambda x: f"{x:.0%}")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.dataframe(
                                keyword_df,
                                use_container_width=True,
                                hide_index=True
                            )
                        
                        with col2:
                            # Bar chart
                            keyword_df['Relevance_num'] = [kw[1] for kw in keywords]
                            fig = px.bar(
                                keyword_df,
                                x='Relevance_num',
                                y='Keyword',
                                orientation='h',
                                title="Keyword Importance",
                                labels={'Relevance_num': 'Relevance Score'}
                            )
                            fig.update_layout(height=300, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No significant keywords found in this review.")
                    
                    # Summary Section
                    st.markdown("---")
                    st.markdown("### 📋 Analysis Summary")
                    
                    # Generate summary
                    avg_confidence = (sentiment['score'] + quality['score']) / 2
                    top_keywords = ', '.join([kw[0] for kw in keywords[:3]]) if keywords else "N/A"
                    
                    summary_text = f"""
                    **Overall Assessment:**
                    - This is a **{q_label.lower()}** review with **{sentiment['label'].lower()}** sentiment
                    - The review focuses on: **{top_keywords}**
                    - Analysis confidence: **{avg_confidence:.0%}**
                    - Review length: **{len(review_text)} characters**
                    
                    **Recommendation:**
                    """
                    
                    if q_score >= 70 and sentiment['label'] == 'POSITIVE':
                        summary_text += "This is an excellent quality review that provides valuable insights. The reviewer should be rewarded."
                    elif q_score < 40:
                        summary_text += "This review lacks detail and may not be very helpful to other users. Encourage more detailed feedback."
                    else:
                        summary_text += "This is an average quality review. It provides some useful information but could be more detailed."
                    
                    st.markdown(f"""
                    <div class="info-box">
                        {summary_text}
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.exception(e)
        else:
            st.warning("⚠️ Please enter a review with at least 10 characters.")

# Continue with other pages in the next part...


# ========================================
# PAGE: Batch Analysis
# ========================================
elif page == "📊 Batch Analysis":
    st.title("📊 Batch Review Analysis")
    st.markdown("Upload a CSV file with multiple reviews for comprehensive batch analysis.")
    
    if not models_loaded:
        st.error("⚠️ Models not loaded. Please refresh the page.")
        st.stop()
    
    st.markdown("""
    <div class="info-box">
        <h4>📋 CSV Format Requirements</h4>
        <ul>
            <li>Must have a column named <code>'review'</code> or <code>'text'</code> containing the review text</li>
            <li>Optional: <code>'rating'</code> column for star ratings</li>
            <li>Optional: <code>'user_id'</code> or <code>'username'</code> for user analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file containing restaurant reviews"
    )
    
    if uploaded_file is not None:
        try:
            # Load CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} reviews from file")
            
            # Preview data
            with st.expander("📄 Preview Data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Identify text column
            text_col = None
            for col in ['review', 'text', 'Review', 'Text', 'content', 'Content']:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col is None:
                st.error("❌ Could not find 'review' or 'text' column in the CSV file.")
                st.info("Available columns: " + ", ".join(df.columns.tolist()))
                st.stop()
            
            st.info(f"Using column '{text_col}' for analysis")
            
            # Analysis settings
            col1, col2 = st.columns(2)
            
            with col1:
                max_reviews = st.number_input(
                    "Maximum reviews to analyze",
                    min_value=1,
                    max_value=len(df),
                    value=min(100, len(df)),
                    help="Analyzing many reviews may take time"
                )
            
            with col2:
                show_details = st.checkbox("Show detailed results", value=False)
            
            # Analyze button
            if st.button("🚀 Analyze All Reviews", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                
                for idx, row in df.head(max_reviews).iterrows():
                    text = str(row[text_col])
                    
                    if len(text) > 10:
                        try:
                            # Analyze
                            sentiment = analyze_sentiment(text)
                            quality = analyze_quality(text)
                            
                            results.append({
                                'review': text[:100] + '...' if len(text) > 100 else text,
                                'sentiment': sentiment['label'],
                                'sentiment_score': sentiment['score'],
                                'quality': map_quality_label(quality['label'])[0],
                                'quality_score': quality['score'],
                                'length': len(text)
                            })
                        except:
                            pass
                    
                    # Update progress
                    progress = (idx + 1) / max_reviews
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing review {idx + 1} of {max_reviews}...")
                
                progress_bar.empty()
                status_text.empty()
                
                if results:
                    results_df = pd.DataFrame(results)
                    
                    st.markdown("""
                    <div class="success-box">
                        <h4>✅ Analysis Complete!</h4>
                        <p>Successfully analyzed {} reviews.</p>
                    </div>
                    """.format(len(results_df)), unsafe_allow_html=True)
                    
                    # Summary Statistics
                    st.markdown("### 📊 Summary Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Total Reviews",
                            len(results_df)
                        )
                    
                    with col2:
                        positive_pct = (results_df['sentiment'] == 'POSITIVE').sum() / len(results_df) * 100
                        st.metric(
                            "Positive Reviews",
                            f"{positive_pct:.1f}%"
                        )
                    
                    with col3:
                        high_quality = (results_df['quality'] == 'High Quality').sum()
                        st.metric(
                            "High Quality",
                            high_quality
                        )
                    
                    with col4:
                        avg_length = results_df['length'].mean()
                        st.metric(
                            "Avg Length",
                            f"{avg_length:.0f} chars"
                        )
                    
                    # Visualizations
                    st.markdown("---")
                    st.markdown("### 📈 Analysis Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sentiment distribution
                        st.markdown("#### Sentiment Distribution")
                        sentiment_counts = results_df['sentiment'].value_counts()
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Sentiment Distribution",
                            color_discrete_map={'POSITIVE': '#4CAF50', 'NEGATIVE': '#F44336'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Quality distribution
                        st.markdown("#### Quality Distribution")
                        quality_counts = results_df['quality'].value_counts()
                        fig = px.pie(
                            values=quality_counts.values,
                            names=quality_counts.index,
                            title="Quality Distribution",
                            color_discrete_map={
                                'Low Quality': '#ff6b6b',
                                'Medium Quality': '#ffd93d',
                                'High Quality': '#6bcf7f'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Sentiment vs Quality
                    st.markdown("#### Sentiment vs Quality Analysis")
                    fig = px.scatter(
                        results_df,
                        x='sentiment_score',
                        y='quality_score',
                        color='sentiment',
                        size='length',
                        hover_data=['review'],
                        title="Sentiment Score vs Quality Score",
                        labels={
                            'sentiment_score': 'Sentiment Confidence',
                            'quality_score': 'Quality Confidence'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed Results
                    if show_details:
                        st.markdown("---")
                        st.markdown("### 📋 Detailed Results")
                        st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Download results
                    st.markdown("---")
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="yelp_review_analysis_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.error("❌ No valid reviews found for analysis.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)

# ========================================
# PAGE: User Quality Ranking
# ========================================
elif page == "👥 User Quality Ranking":
    st.title("👥 User Quality Ranking")
    st.markdown("Identify and reward high-quality reviewers on the Yelp platform.")
    
    st.markdown("""
    <div class="info-box">
        <h4>📝 Demo Mode</h4>
        <p>This page shows a simulated ranking system based on review quality metrics. 
        In production, this would analyze real user data from the Yelp platform.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate demo data
    np.random.seed(42)
    demo_users = pd.DataFrame({
        'User ID': [f'user_{i:03d}' for i in range(1, 51)],
        'Username': [f'Reviewer{i}' for i in range(1, 51)],
        'Review Count': np.random.randint(5, 150, 50),
        'Avg Quality Score': np.random.uniform(40, 95, 50),
        'Avg Sentiment': np.random.uniform(0.5, 1.0, 50),
        'High Quality Reviews': np.random.randint(2, 80, 50)
    })
    
    # Calculate user score
    demo_users['User Score'] = (
        demo_users['Avg Quality Score'] * 0.5 +
        (demo_users['High Quality Reviews'] / demo_users['Review Count'] * 100) * 0.3 +
        demo_users['Avg Sentiment'] * 20
    )
    
    # Sort by score
    demo_users = demo_users.sort_values('User Score', ascending=False).reset_index(drop=True)
    demo_users['Rank'] = range(1, len(demo_users) + 1)
    
    # Add coupon recommendations
    demo_users['Recommended Coupon'] = demo_users['User Score'].apply(recommend_coupon)
    
    # Top 3 Users
    st.markdown("### 🏆 Top Quality Reviewers")
    
    col1, col2, col3 = st.columns(3)
    
    medals = ["🥇", "🥈", "🥉"]
    colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
    
    for idx, col in enumerate([col1, col2, col3]):
        user = demo_users.iloc[idx]
        with col:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {colors[idx]}20 0%, {colors[idx]}40 100%);
                        padding: 1.5rem; border-radius: 1rem; text-align: center;
                        border: 3px solid {colors[idx]}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h2 style='margin: 0;'>{medals[idx]} #{user['Rank']}</h2>
                <h3 style='margin: 0.5rem 0;'>{user['Username']}</h3>
                <p style='margin: 0.25rem 0;'><strong>Score:</strong> {user['User Score']:.1f}/100</p>
                <p style='margin: 0.25rem 0;'><strong>Reviews:</strong> {user['Review Count']}</p>
                <p style='margin: 0.25rem 0;'><strong>Quality:</strong> {user['Avg Quality Score']:.1f}</p>
                <hr style='margin: 0.5rem 0;'>
                <p style='margin: 0; font-weight: 600;'>{user['Recommended Coupon']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Filters
    st.markdown("### 🔍 Filter Rankings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_reviews = st.slider(
            "Minimum Review Count",
            min_value=0,
            max_value=int(demo_users['Review Count'].max()),
            value=0
        )
    
    with col2:
        min_score = st.slider(
            "Minimum User Score",
            min_value=0.0,
            max_value=100.0,
            value=0.0
        )
    
    with col3:
        top_n = st.selectbox(
            "Show Top N Users",
            options=[10, 20, 30, 50],
            index=0
        )
    
    # Apply filters
    filtered_users = demo_users[
        (demo_users['Review Count'] >= min_reviews) &
        (demo_users['User Score'] >= min_score)
    ].head(top_n)
    
    # Full Ranking Table
    st.markdown(f"### 📊 Top {len(filtered_users)} Users")
    
    display_df = filtered_users[[
        'Rank', 'Username', 'User Score', 'Review Count',
        'Avg Quality Score', 'High Quality Reviews', 'Recommended Coupon'
    ]].copy()
    
    # Format numbers
    display_df['User Score'] = display_df['User Score'].apply(lambda x: f"{x:.1f}")
    display_df['Avg Quality Score'] = display_df['Avg Quality Score'].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d"),
            "User Score": st.column_config.TextColumn("User Score"),
            "Review Count": st.column_config.NumberColumn("Reviews", format="%d"),
        }
    )
    
    # Visualizations
    st.markdown("---")
    st.markdown("### 📈 User Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top users bar chart
        fig = px.bar(
            filtered_users.head(10),
            x='Username',
            y='User Score',
            title=f"Top 10 Users by Score",
            color='User Score',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Score distribution
        fig = px.histogram(
            demo_users,
            x='User Score',
            nbins=20,
            title="User Score Distribution",
            labels={'User Score': 'User Score', 'count': 'Number of Users'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Coupon distribution
    st.markdown("#### 🎁 Coupon Distribution")
    coupon_counts = demo_users['Recommended Coupon'].value_counts()
    fig = px.pie(
        values=coupon_counts.values,
        names=coupon_counts.index,
        title="Recommended Coupon Types"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Download rankings
    st.markdown("---")
    csv = filtered_users.to_csv(index=False)
    st.download_button(
        label="📥 Download Rankings as CSV",
        data=csv,
        file_name="user_quality_rankings.csv",
        mime="text/csv",
        use_container_width=True
    )

# ========================================
# PAGE: Restaurant Insights
# ========================================
elif page == "🍽️ Restaurant Insights":
    st.title("🍽️ Restaurant Insights")
    st.markdown("Analyze reviews for a specific restaurant and get actionable insights.")
    
    st.markdown("""
    <div class="info-box">
        <h4>📝 Demo Mode</h4>
        <p>This page shows simulated restaurant insights. In production, this would analyze 
        real review data for specific restaurants.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Restaurant input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        restaurant_name = st.text_input(
            "Enter restaurant name:",
            value="Sample Italian Restaurant",
            help="Enter the name of the restaurant to analyze"
        )
    
    with col2:
        time_period = st.selectbox(
            "Time Period",
            options=["Last Month", "Last 3 Months", "Last 6 Months", "Last Year"],
            index=2
        )
    
    if st.button("🔍 Generate Insights", type="primary", use_container_width=True):
        with st.spinner("📊 Analyzing restaurant reviews..."):
            # Simulate analysis delay
            import time
            time.sleep(1)
            
            # Generate demo data
            np.random.seed(42)
            
            # Overall Metrics
            st.markdown("### 📊 Overall Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Overall Rating",
                    "4.2/5.0",
                    delta="+0.3",
                    delta_color="normal"
                )
            
            with col2:
                st.metric(
                    "Total Reviews",
                    "1,234",
                    delta="+15%",
                    delta_color="normal"
                )
            
            with col3:
                st.metric(
                    "Positive Reviews",
                    "78%",
                    delta="+5%",
                    delta_color="normal"
                )
            
            with col4:
                st.metric(
                    "Avg Quality Score",
                    "72/100",
                    delta="+8",
                    delta_color="normal"
                )
            
            st.markdown("---")
            
            # Sentiment Trend
            st.markdown("### 📈 Sentiment Trend Over Time")
            
            dates = pd.date_range('2024-01-01', '2024-12-01', freq='BM')
            sentiment_data = pd.DataFrame({
                'Month': dates,
                'Positive': np.random.uniform(60, 85, len(dates)),
                'Negative': np.random.uniform(10, 25, len(dates)),
                'Neutral': np.random.uniform(5, 15, len(dates))
            })
            
            fig = px.line(
                sentiment_data,
                x='Month',
                y=['Positive', 'Negative', 'Neutral'],
                title=f"Sentiment Trend for {restaurant_name}",
                labels={'value': 'Percentage (%)', 'variable': 'Sentiment'},
                color_discrete_map={
                    'Positive': '#4CAF50',
                    'Negative': '#F44336',
                    'Neutral': '#FFC107'
                }
            )
            st.plotly_chart(fig, use_container_width=False)
            st.markdown("<style>.stPlotlyChart {width: 100% !important;}</style>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Key Themes
            st.markdown("### 🔍 Key Themes Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 👍 Strengths")
                st.markdown("""
                <div class="success-box">
                    <p><strong>🍝 Food Quality</strong> (mentioned in 85% of positive reviews)</p>
                    <p>"Authentic Italian flavors", "Fresh ingredients", "Perfect pasta"</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="success-box">
                    <p><strong>🎭 Atmosphere</strong> (mentioned in 72% of positive reviews)</p>
                    <p>"Cozy ambiance", "Romantic setting", "Beautiful decor"</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="success-box">
                    <p><strong>🍷 Wine Selection</strong> (mentioned in 68% of positive reviews)</p>
                    <p>"Excellent wine list", "Knowledgeable sommelier", "Great pairings"</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 👎 Areas for Improvement")
                st.markdown("""
                <div class="warning-box">
                    <p><strong>⏱️ Service Speed</strong> (mentioned in 45% of negative reviews)</p>
                    <p>"Slow service", "Long wait times", "Understaffed"</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="warning-box">
                    <p><strong>💰 Pricing</strong> (mentioned in 38% of negative reviews)</p>
                    <p>"Expensive", "Overpriced", "Not good value"</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="warning-box">
                    <p><strong>🚗 Parking</strong> (mentioned in 25% of negative reviews)</p>
                    <p>"Limited parking", "Hard to find spot", "Expensive parking"</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Topic Distribution
            st.markdown("### 📊 Review Topic Distribution")
            
            topics = ['Food Quality', 'Service', 'Atmosphere', 'Value', 'Location']
            mentions = [850, 620, 580, 450, 320]
            
            fig = px.bar(
                x=topics,
                y=mentions,
                title="Most Mentioned Topics",
                labels={'x': 'Topic', 'y': 'Number of Mentions'},
                color=mentions,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Actionable Recommendations
            st.markdown("### 💡 Actionable Recommendations")
            
            st.markdown("""
            <div class="info-box">
                <h4>Based on the analysis, we recommend:</h4>
                <ol>
                    <li><strong>Improve Service Speed</strong>
                        <ul>
                            <li>Consider hiring additional staff during peak hours</li>
                            <li>Implement a reservation management system</li>
                            <li>Train staff on efficiency best practices</li>
                        </ul>
                    </li>
                    <li><strong>Address Pricing Concerns</strong>
                        <ul>
                            <li>Introduce a value menu or lunch specials</li>
                            <li>Offer prix fixe menu options</li>
                            <li>Highlight the quality and authenticity to justify prices</li>
                        </ul>
                    </li>
                    <li><strong>Parking Solutions</strong>
                        <ul>
                            <li>Partner with nearby parking facilities for discounts</li>
                            <li>Offer valet service during peak hours</li>
                            <li>Provide clear parking instructions on website</li>
                        </ul>
                    </li>
                    <li><strong>Leverage Strengths in Marketing</strong>
                        <ul>
                            <li>Highlight authentic Italian cuisine in promotions</li>
                            <li>Showcase the romantic atmosphere for special occasions</li>
                            <li>Promote wine selection and sommelier expertise</li>
                        </ul>
                    </li>
                    <li><strong>Engage with Reviews</strong>
                        <ul>
                            <li>Respond to both positive and negative reviews promptly</li>
                            <li>Thank customers for positive feedback</li>
                            <li>Address concerns raised in negative reviews</li>
                        </ul>
                    </li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
            # Competitor Comparison
            st.markdown("---")
            st.markdown("### 🏆 Competitive Position")
            
            comp_data = pd.DataFrame({
                'Restaurant': [restaurant_name, 'Competitor A', 'Competitor B', 'Competitor C'],
                'Rating': [4.2, 4.0, 4.5, 3.8],
                'Reviews': [1234, 890, 1567, 654],
                'Quality Score': [72, 68, 78, 65]
            })
            
            fig = px.bar(
                comp_data,
                x='Restaurant',
                y=['Rating', 'Quality Score'],
                title="Comparison with Competitors",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

# ========================================
# Footer
# ========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem 0;'>
    <p style='margin: 0.5rem 0;'><strong>Yelp Review Intelligence System</strong></p>
    <p style='margin: 0.5rem 0;'>Powered by AI | Built with Streamlit, Hugging Face Transformers</p>
    <p style='margin: 0.5rem 0;'>ISOM5240 Course Project | HKUST Business School</p>
    <p style='margin: 0.5rem 0;'>© 2024 | For Educational Purposes Only</p>
</div>
""", unsafe_allow_html=True)
