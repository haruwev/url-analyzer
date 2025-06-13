#!/usr/bin/env python3
"""
Standalone Streamlit App for URL Analysis
All-in-one version suitable for Streamlit Community Cloud deployment
"""

import streamlit as st
import asyncio
import time
import json
import re
from typing import Optional, List, Dict, Set
from urllib.parse import urljoin, urlparse
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="URL Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DeepCrawler:
    """Deep crawling functionality for comprehensive site analysis"""
    
    def __init__(self, max_concurrent: int = 3, max_pages: int = 10, max_depth: int = 2):
        self.max_concurrent = max_concurrent
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.visited_urls: Set[str] = set()
        self.found_pages: List[Dict] = []
        self.crawl_queue: List[str] = []
        self.start_time = time.time()
        self.progress_callback = None
    
    def set_progress_callback(self, callback):
        """Set callback for progress updates"""
        self.progress_callback = callback
    
    def report_progress(self, message: str):
        """Report progress"""
        if self.progress_callback:
            elapsed = time.time() - self.start_time
            progress = {
                "pages_found": len(self.found_pages),
                "pages_visited": len(self.visited_urls),
                "max_pages": self.max_pages,
                "max_depth": self.max_depth,
                "elapsed_time": round(elapsed, 2),
                "queue_size": len(self.crawl_queue),
                "message": message
            }
            self.progress_callback(progress)
        
    def is_valid_url(self, url: str, base_domain: str) -> bool:
        """Check if URL is valid for crawling"""
        try:
            parsed = urlparse(url)
            base_parsed = urlparse(base_domain)
            
            # Same domain only
            if parsed.netloc != base_parsed.netloc:
                return False
                
            # Skip common non-content URLs
            skip_patterns = [
                r'\.pdf$', r'\.doc$', r'\.zip$', r'\.exe$',
                r'/admin/', r'/login/', r'/logout/', r'/cart/',
                r'\.(css|js|ico|png|jpg|jpeg|gif|svg)$'
            ]
            
            for pattern in skip_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    return False
                    
            return True
        except:
            return False
    
    def extract_links(self, html: str, base_url: str) -> Set[str]:
        """Extract links from HTML content"""
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)
            
            if self.is_valid_url(absolute_url, base_url):
                links.add(absolute_url)
                
        return links
    
    async def crawl_page(self, url: str, depth: int = 0) -> Optional[Dict]:
        """Crawl a single page and extract content"""
        if url in self.visited_urls or depth > self.max_depth or len(self.found_pages) >= self.max_pages:
            return None
            
        async with self.semaphore:
            try:
                self.visited_urls.add(url)
                self.report_progress(f"Crawling: {url} (depth {depth})")
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (compatible; URL-Analyzer/1.0)'
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=30, headers=headers) as response:
                        if response.status != 200:
                            self.report_progress(f"Failed to fetch {url}: HTTP {response.status}")
                            return None
                        
                        html = await response.text()
                        self.report_progress(f"Processing content from {url}")
                        
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Extract page info
                title_tag = soup.find('title')
                title = title_tag.get_text().strip() if title_tag else "No title"
                
                # Extract meta description
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                description = meta_desc.get('content', '') if meta_desc else ''
                
                # Extract text content
                text_content = soup.get_text()
                lines = (line.strip() for line in text_content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                content = ' '.join(chunk for chunk in chunks if chunk)
                
                # Extract images with comprehensive methods
                image_urls = self.extract_comprehensive_images(soup, url)
                
                # Extract links for further crawling
                links = self.extract_links(html, url)
                
                page_data = {
                    "url": url,
                    "path": urlparse(url).path or "/",
                    "title": title,
                    "description": description,
                    "content": content[:10000],  # Limit content length
                    "image_urls": list(image_urls),
                    "internal_links": list(links),
                    "depth": depth,
                    "word_count": len(content.split()),
                    "image_count": len(image_urls)
                }
                
                self.found_pages.append(page_data)
                
                # Update crawl queue for next depth level
                new_links = [link for link in links if link not in self.visited_urls]
                self.crawl_queue.extend(new_links[:10])  # Add up to 10 new links
                
                self.report_progress(f"Completed {url} - Found {len(image_urls)} images, {len(new_links)} new links")
                
                # Crawl linked pages if depth allows
                if depth < self.max_depth and len(self.found_pages) < self.max_pages:
                    crawl_tasks = []
                    for link in new_links[:3]:  # Limit concurrent crawls for Streamlit
                        if link not in self.visited_urls:
                            crawl_tasks.append(self.crawl_page(link, depth + 1))
                    
                    if crawl_tasks:
                        self.report_progress(f"Starting crawl of {len(crawl_tasks)} linked pages from {url}")
                        await asyncio.gather(*crawl_tasks, return_exceptions=True)
                
                return page_data
                
            except Exception as e:
                self.report_progress(f"Error crawling {url}: {e}")
                return None
    
    def extract_comprehensive_images(self, soup: BeautifulSoup, base_url: str) -> Set[str]:
        """Extract images using multiple methods"""
        images = set()
        
        # 1. Standard img tags
        for img in soup.find_all('img'):
            src = img.get('src') or img.get('data-src') or img.get('data-lazy-src') or img.get('data-original')
            if src:
                images.add(urljoin(base_url, src))
        
        # 2. CSS background images
        for elem in soup.find_all(attrs={'style': True}):
            style = elem['style']
            bg_matches = re.findall(r'background-image:\s*url\(["\']?([^"\'\s]+)["\']?\)', style)
            for match in bg_matches:
                images.add(urljoin(base_url, match))
        
        # 3. Picture elements
        for picture in soup.find_all('picture'):
            for source in picture.find_all('source'):
                srcset = source.get('srcset')
                if srcset:
                    urls = re.findall(r'([^\s,]+)', srcset)
                    for url in urls:
                        if not url.endswith('w') and not url.endswith('x'):  # Skip descriptors
                            images.add(urljoin(base_url, url))
        
        # 4. Lazy loading attributes
        for elem in soup.find_all(attrs={'data-src': True}):
            images.add(urljoin(base_url, elem['data-src']))
        
        # Filter valid image URLs
        valid_images = set()
        for img_url in images:
            if self.is_valid_image_url(img_url):
                valid_images.add(img_url)
        
        return valid_images
    
    def is_valid_image_url(self, url: str) -> bool:
        """Check if URL is a valid image"""
        try:
            parsed = urlparse(url)
            path = parsed.path.lower()
            
            # Check file extension
            image_exts = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.bmp', '.tiff']
            if any(path.endswith(ext) for ext in image_exts):
                return True
            
            # Skip data URIs
            if url.startswith('data:'):
                return False
                
            # Skip very short URLs
            if len(url) < 10:
                return False
                
            return True
        except:
            return False
    
    async def crawl_site(self, start_url: str) -> List[Dict]:
        """Crawl entire site starting from given URL"""
        self.report_progress(f"Starting site crawl from {start_url}")
        await self.crawl_page(start_url, 0)
        self.report_progress(f"Site crawl completed - {len(self.found_pages)} pages found")
        return self.found_pages

async def analyze_single_page(url: str) -> dict:
    """Extract basic content from a single URL"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=30) as response:
                if response.status != 200:
                    return {
                        "path": "/",
                        "title": "Error",
                        "content": f"HTTP {response.status}: Could not fetch content",
                        "image_urls": [],
                        "word_count": 0
                    }
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract title
                title_tag = soup.find('title')
                title = title_tag.get_text().strip() if title_tag else "No title"
                
                # Extract text content
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text_content = soup.get_text()
                lines = (line.strip() for line in text_content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                content = ' '.join(chunk for chunk in chunks if chunk)
                
                # Extract image URLs
                image_urls = []
                for img in soup.find_all('img'):
                    src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                    if src:
                        absolute_url = urljoin(url, src)
                        image_urls.append(absolute_url)
                
                image_urls = list(set(image_urls))
                
                return {
                    "url": url,
                    "path": urlparse(url).path or "/",
                    "title": title,
                    "content": content[:5000],
                    "image_urls": image_urls,
                    "word_count": len(content.split()),
                    "depth": 0
                }
                
    except Exception as e:
        return {
            "url": url,
            "path": "/",
            "title": "Error",
            "content": f"Error extracting content: {str(e)}",
            "image_urls": [],
            "word_count": 0,
            "depth": 0
        }

def display_progress(progress_data: Dict):
    """Display progress information"""
    if not progress_data:
        return
    
    # Progress metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pages Found", progress_data.get("pages_found", 0), 
                 delta=f"/ {progress_data.get('max_pages', 1)}")
    with col2:
        st.metric("Time Elapsed", f"{progress_data.get('elapsed_time', 0)}s")
    with col3:
        visited = progress_data.get("pages_visited", 0)
        st.metric("Pages Visited", visited)
    
    # Progress bar
    pages_found = progress_data.get("pages_found", 0)
    max_pages = progress_data.get("max_pages", 1)
    progress_percent = min(int((pages_found / max_pages) * 100), 90)
    st.progress(progress_percent)
    
    # Current activity
    message = progress_data.get("message", "Processing...")
    st.info(f"🔄 {message}")
    
    # Queue info
    queue_size = progress_data.get("queue_size", 0)
    if queue_size > 0:
        st.write(f"📋 Queue: {queue_size} URLs pending")

def display_analysis_results(result_data: List[Dict], analysis_time: float):
    """Display analysis results"""
    if not result_data:
        st.warning("No data to display")
        return
    
    # Summary metrics
    total_pages = len(result_data)
    total_images = sum(len(item.get("image_urls", [])) for item in result_data)
    total_words = sum(item.get("word_count", 0) for item in result_data)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pages", total_pages)
    with col2:
        st.metric("Total Images", total_images)
    with col3:
        st.metric("Total Words", f"{total_words:,}")
    with col4:
        st.metric("Analysis Time", f"{analysis_time:.2f}s")
    
    st.divider()
    
    # Data display tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Page Content", "🖼️ Images", "📊 Summary Table", "📈 Analytics"])
    
    with tab1:
        st.subheader("Page Content Analysis")
        
        # Add filtering and sorting options
        col1, col2, col3 = st.columns(3)
        with col1:
            depth_filter = st.selectbox("Filter by Depth", [f"All"] + [f"Depth {i}" for i in range(5)])
        with col2:
            sort_by = st.selectbox("Sort by", ["URL", "Title", "Word Count", "Image Count", "Depth"])
        with col3:
            show_full_content = st.checkbox("Show Full Content")
        
        # Filter and sort data
        filtered_data = result_data.copy()
        if depth_filter != "All":
            depth_num = int(depth_filter.split()[-1])
            filtered_data = [item for item in filtered_data if item.get('depth', 0) == depth_num]
        
        # Sort data
        if sort_by == "Word Count":
            filtered_data.sort(key=lambda x: x.get('word_count', 0), reverse=True)
        elif sort_by == "Image Count":
            filtered_data.sort(key=lambda x: len(x.get('image_urls', [])), reverse=True)
        elif sort_by == "Depth":
            filtered_data.sort(key=lambda x: x.get('depth', 0))
        elif sort_by == "Title":
            filtered_data.sort(key=lambda x: x.get('title', ''))
        else:  # URL
            filtered_data.sort(key=lambda x: x.get('url', ''))
        
        st.write(f"Showing {len(filtered_data)} of {len(result_data)} pages")
        
        for i, item in enumerate(filtered_data):
            page_url = item.get('url', 'Unknown URL')
            page_title = item.get('title', 'No title')
            depth = item.get('depth', 0)
            word_count = item.get('word_count', 0)
            image_count = len(item.get('image_urls', []))
            
            with st.expander(f"📄 {page_title} (Depth {depth})", expanded=False):
                # Page metrics
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Depth", depth)
                with metric_col2:
                    st.metric("Words", word_count)
                with metric_col3:
                    st.metric("Images", image_count)
                with metric_col4:
                    st.metric("Links", len(item.get('internal_links', [])))
                
                # Page details
                st.write("**URL:**", page_url)
                st.write("**Path:**", item.get('path', 'No path'))
                
                description = item.get('description', '')
                if description:
                    st.write("**Description:**", description)
                
                content = item.get('content', '')
                if content:
                    st.write("**Content:**")
                    if show_full_content:
                        st.text_area("", content, height=200, key=f"content_full_{i}", disabled=True)
                    else:
                        preview = content[:500] + "..." if len(content) > 500 else content
                        st.text_area("", preview, height=150, key=f"content_preview_{i}", disabled=True)
                
                # Image preview section
                image_urls = item.get('image_urls', [])
                if image_urls:
                    st.write("**Images Found:**")
                    img_cols = st.columns(min(len(image_urls), 4))
                    for j, img_url in enumerate(image_urls[:4]):
                        with img_cols[j]:
                            try:
                                st.image(img_url, caption=f"Image {j+1}", use_container_width=True)
                            except:
                                st.text(f"Image {j+1}: {img_url}")
                    if len(image_urls) > 4:
                        st.write(f"... and {len(image_urls) - 4} more images")
    
    with tab2:
        st.subheader("Image Gallery")
        
        all_images = []
        for item in result_data:
            path = item.get('path', 'Unknown path')
            for img_url in item.get('image_urls', []):
                all_images.append({"path": path, "url": img_url})
        
        if all_images:
            for i, item in enumerate(result_data):
                image_urls = item.get('image_urls', [])
                if image_urls:
                    st.write(f"**{item.get('path', f'Page {i+1}')}** ({len(image_urls)} images)")
                    
                    cols = st.columns(min(len(image_urls), 4))
                    for j, img_url in enumerate(image_urls[:8]):
                        with cols[j % 4]:
                            try:
                                st.image(img_url, caption=f"Image {j+1}", use_container_width=True)
                            except:
                                st.text(f"Image {j+1}: {img_url}")
                    
                    if len(image_urls) > 8:
                        st.text(f"... and {len(image_urls) - 8} more images")
                    st.divider()
        else:
            st.info("No images found in the analyzed content")
    
    with tab3:
        st.subheader("Summary Table")
        
        # Create summary dataframe
        summary_data = []
        for i, item in enumerate(result_data):
            summary_data.append({
                "Page": i + 1,
                "Path": item.get('path', 'Unknown'),
                "Title": item.get('title', 'No title')[:50] + "..." if len(item.get('title', '')) > 50 else item.get('title', 'No title'),
                "Content Length": len(item.get('content', '')),
                "Images Count": len(item.get('image_urls', [])),
                "Word Count": item.get('word_count', 0),
                "Depth": item.get('depth', 0)
            })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)
            
            # Download options
            st.subheader("Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                json_data = json.dumps(result_data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download as JSON",
                    data=json_data,
                    file_name=f"url_analysis_{int(time.time())}.json",
                    mime="application/json"
                )
            
            with col2:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download Summary as CSV",
                    data=csv_data,
                    file_name=f"url_analysis_summary_{int(time.time())}.csv",
                    mime="text/csv"
                )
    
    with tab4:
        st.subheader("Content Analytics")
        
        if result_data:
            # Content statistics
            avg_words = total_words / len(result_data) if result_data else 0
            avg_images = total_images / len(result_data) if result_data else 0
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Words", f"{total_words:,}")
            with col2:
                st.metric("Avg Words/Page", f"{avg_words:.0f}")
            with col3:
                st.metric("Total Images", total_images)
            with col4:
                st.metric("Avg Images/Page", f"{avg_images:.1f}")
            
            # Word count distribution
            word_counts = [item.get('word_count', 0) for item in result_data]
            
            if max(word_counts) > 0:
                fig_words = px.histogram(x=word_counts, title="Word Count Distribution", 
                                       labels={'x': 'Words per Page', 'y': 'Number of Pages'})
                st.plotly_chart(fig_words, use_container_width=True)
            
            # Top content pages
            st.write("**Most Content-Rich Pages:**")
            content_sorted = sorted(result_data, key=lambda x: x.get('word_count', 0), reverse=True)
            for item in content_sorted[:5]:
                title = item.get('title', 'No title')
                words = item.get('word_count', 0)
                images = len(item.get('image_urls', []))
                st.write(f"• **{title}** - {words:,} words, {images} images")

def main():
    st.title("🔍 URL Analyzer")
    st.write("Enter a URL to analyze its content and extract images")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("🔧 Analysis Settings")
        
        # Deep crawling settings
        st.subheader("Deep Crawling")
        follow_links = st.checkbox("Follow Links (Deep Crawl)", value=True, 
                                  help="Enable deep crawling to analyze multiple pages")
        
        if follow_links:
            max_pages = st.slider("Max Pages", min_value=1, max_value=20, value=5, 
                                help="Maximum number of pages to analyze")
            max_depth = st.slider("Max Depth", min_value=1, max_value=3, value=2, 
                                help="Maximum crawling depth")
        else:
            max_pages = 1
            max_depth = 0
        
        st.divider()
        
        # Show current settings
        st.subheader("📋 Current Settings")
        st.write(f"**Deep Crawl:** {'✅ Enabled' if follow_links else '❌ Single Page'}")
        if follow_links:
            st.write(f"**Max Pages:** {max_pages}")
            st.write(f"**Max Depth:** {max_depth}")
    
    # Main form
    with st.form("url_form"):
        url_input = st.text_input(
            "URL to Analyze",
            placeholder="https://example.com",
            help="Enter the URL you want to analyze"
        )
        
        submit_button = st.form_submit_button("🚀 Start Analysis", use_container_width=True)
    
    if submit_button and url_input:
        if not url_input.startswith(('http://', 'https://')):
            st.error("Please enter a valid URL starting with http:// or https://")
            return
        
        # Initialize session state for results
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        
        # Analysis
        start_time = time.time()
        
        with st.spinner("Starting analysis..."):
            st.info(f"⚙️ Analysis Settings: Max Pages: {max_pages}, Max Depth: {max_depth}, Follow Links: {follow_links}")
            
            # Create progress containers
            progress_container = st.empty()
            
            try:
                if follow_links and max_depth > 0:
                    # Deep crawling
                    crawler = DeepCrawler(
                        max_concurrent=2,  # Reduced for Streamlit Cloud
                        max_pages=max_pages, 
                        max_depth=max_depth
                    )
                    
                    # Set up progress callback
                    def update_progress(progress_data):
                        with progress_container.container():
                            display_progress(progress_data)
                    
                    crawler.set_progress_callback(update_progress)
                    
                    # Run deep crawling
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        collected_data = loop.run_until_complete(crawler.crawl_site(url_input))
                    finally:
                        loop.close()
                else:
                    # Single page analysis
                    with progress_container.container():
                        st.info("🔄 Analyzing single page...")
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        content_data = loop.run_until_complete(analyze_single_page(url_input))
                        collected_data = [content_data] if content_data else []
                    finally:
                        loop.close()
                
                end_time = time.time()
                analysis_time = end_time - start_time
                
                if collected_data:
                    # Save results to session state
                    st.session_state.analysis_results = {
                        'data': collected_data,
                        'analysis_time': analysis_time
                    }
                    
                    progress_container.empty()
                    st.success("🎉 Analysis completed successfully!")
                    display_analysis_results(collected_data, analysis_time)
                else:
                    st.warning("No content found for the given URL")
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
    
    # Display previous results if available
    if hasattr(st.session_state, 'analysis_results') and st.session_state.analysis_results:
        if not submit_button:  # Only show if not currently analyzing
            st.divider()
            st.subheader("📋 Previous Analysis Results")
            st.info("Showing results from previous analysis. You can still download the data below.")
            display_analysis_results(
                st.session_state.analysis_results['data'], 
                st.session_state.analysis_results['analysis_time']
            )
    
    # Example URLs section
    st.divider()
    st.subheader("💡 Example URLs")
    
    example_urls = [
        "https://www.cosmowater.com/",
        "https://example.com",
        "https://httpbin.org/html"
    ]
    
    cols = st.columns(len(example_urls))
    for i, example_url in enumerate(example_urls):
        with cols[i]:
            if st.button(f"Try: {example_url}", key=f"example_{i}"):
                st.session_state.url_input = example_url
                st.rerun()

if __name__ == "__main__":
    main()