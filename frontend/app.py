import streamlit as st
import requests
import json
import time
import os
from typing import Optional, Dict, Any
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="URL Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8002")

def check_backend_health() -> bool:
    """Check if backend is available"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_analysis(url: str, max_pages: int = 10, max_depth: int = 2, follow_links: bool = True, match_patterns: Optional[list] = None, content_selector: Optional[str] = None) -> Dict[str, Any]:
    """Start URL analysis"""
    payload = {
        "url": url,
        "max_pages": max_pages,
        "max_depth": max_depth,
        "follow_links": follow_links,
        "match_patterns": match_patterns or [],
        "content_selector": content_selector
    }
    
    try:
        response = requests.post(f"{BACKEND_URL}/analyze", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to start analysis: {str(e)}")
        return {}

def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get task status"""
    try:
        response = requests.get(f"{BACKEND_URL}/status/{task_id}", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get status: {str(e)}")
        return {}

def display_progress(status_data: Dict[str, Any], progress_bar, status_placeholder):
    """Display detailed progress information"""
    status = status_data.get("status", "unknown")
    message = status_data.get("message", "")
    progress = status_data.get("progress", {})
    
    if status == "completed":
        progress_bar.progress(100)
        status_placeholder.success(f"✅ {message}")
        return True
        
    elif status == "failed":
        progress_bar.progress(100)
        status_placeholder.error(f"❌ {message}")
        return True
        
    elif status == "processing":
        if progress:
            pages_found = progress.get("pages_found", 0)
            max_pages = progress.get("max_pages", 1)
            elapsed_time = progress.get("elapsed_time", 0)
            
            # Calculate progress percentage
            progress_percent = min(int((pages_found / max_pages) * 100), 90)  # Cap at 90% until completion
            progress_bar.progress(progress_percent)
            
            # Detailed progress display
            col1, col2, col3 = status_placeholder.columns(3)
            with col1:
                st.metric("Pages Found", pages_found, delta=f"/ {max_pages}")
            with col2:
                st.metric("Time Elapsed", f"{elapsed_time}s")
            with col3:
                visited = progress.get("pages_visited", 0)
                st.metric("Pages Visited", visited)
            
            # Current activity
            st.info(f"🔄 {message}")
            
            # Additional details if available
            if progress.get("queue_size", 0) > 0:
                st.write(f"📋 Queue: {progress['queue_size']} URLs pending")
        else:
            progress_bar.progress(50)
            status_placeholder.info(f"🔄 {message}")
            
    else:  # queued or other
        progress_bar.progress(25)
        status_placeholder.info(f"⏳ {message}")
        
    return False

def display_analysis_results(result_data: Dict[str, Any]):
    """Display analysis results"""
    if not result_data or "data" not in result_data:
        st.warning("No data to display")
        return
    
    data = result_data["data"]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pages", result_data.get("total_pages", 0))
    with col2:
        st.metric("Total Images", result_data.get("total_images", 0))
    with col3:
        st.metric("Analysis Time", f"{result_data.get('analysis_time', 0):.2f}s")
    with col4:
        st.metric("Status", result_data.get("status", "unknown").upper())
    
    st.divider()
    
    # Data display tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Page Content", "🖼️ Images", "📊 Summary Table", "🔗 Site Structure", "📈 Analytics"])
    
    with tab1:
        st.subheader("Page Content Analysis")
        
        # Add filtering and sorting options
        col1, col2, col3 = st.columns(3)
        with col1:
            depth_filter = st.selectbox("Filter by Depth", ["All"] + [f"Depth {i}" for i in range(5)])
        with col2:
            sort_by = st.selectbox("Sort by", ["URL", "Title", "Word Count", "Image Count", "Depth"])
        with col3:
            show_full_content = st.checkbox("Show Full Content")
        
        # Filter and sort data
        filtered_data = data.copy()
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
        
        st.write(f"Showing {len(filtered_data)} of {len(data)} pages")
        
        for i, item in enumerate(filtered_data):
            # Enhanced page header with metrics
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
                col1, col2 = st.columns([3, 1])
                
                with col1:
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
                    else:
                        st.write("*No content available*")
                
                with col2:
                    st.write("**Page Info:**")
                    st.write(f"• Depth: {depth}")
                    st.write(f"• Words: {word_count:,}")
                    st.write(f"• Images: {image_count}")
                    st.write(f"• Internal Links: {len(item.get('internal_links', []))}")
                    
                    # Show some internal links
                    internal_links = item.get('internal_links', [])
                    if internal_links:
                        st.write("**Sample Links:**")
                        for link in internal_links[:3]:
                            st.write(f"• {link}")
                        if len(internal_links) > 3:
                            st.write(f"... and {len(internal_links) - 3} more")
                
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
        for item in data:
            path = item.get('path', 'Unknown path')
            for img_url in item.get('image_urls', []):
                all_images.append({"path": path, "url": img_url})
        
        if all_images:
            # Group images by page
            for i, item in enumerate(data):
                image_urls = item.get('image_urls', [])
                if image_urls:
                    st.write(f"**{item.get('path', f'Page {i+1}')}** ({len(image_urls)} images)")
                    
                    # Display images in columns
                    cols = st.columns(min(len(image_urls), 4))
                    for j, img_url in enumerate(image_urls[:8]):  # Limit to 8 images per page
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
    
    with tab4:
        st.subheader("Site Structure & Link Analysis")
        
        # Site structure visualization
        if data:
            # Group pages by depth
            depth_groups = {}
            for item in data:
                depth = item.get('depth', 0)
                if depth not in depth_groups:
                    depth_groups[depth] = []
                depth_groups[depth].append(item)
            
            st.write("**Site Hierarchy:**")
            for depth in sorted(depth_groups.keys()):
                st.write(f"**Depth {depth}** ({len(depth_groups[depth])} pages):")
                for item in depth_groups[depth][:10]:  # Show first 10 per depth
                    title = item.get('title', 'No title')
                    url = item.get('url', '')
                    st.write(f"{'  ' * depth}• {title}")
                    st.write(f"{'  ' * depth}  📍 {url}")
                if len(depth_groups[depth]) > 10:
                    st.write(f"{'  ' * depth}... and {len(depth_groups[depth]) - 10} more")
                st.write("")
            
            # Link analysis
            st.write("**Internal Link Analysis:**")
            all_links = set()
            total_internal_links = 0
            for item in data:
                internal_links = item.get('internal_links', [])
                total_internal_links += len(internal_links)
                all_links.update(internal_links)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Internal Links", total_internal_links)
            with col2:
                st.metric("Unique URLs Found", len(all_links))
            with col3:
                st.metric("Average Links/Page", f"{total_internal_links/len(data):.1f}" if data else 0)
            
            # Most linked pages
            link_counts = {}
            for item in data:
                for link in item.get('internal_links', []):
                    link_counts[link] = link_counts.get(link, 0) + 1
            
            if link_counts:
                st.write("**Most Referenced Pages:**")
                sorted_links = sorted(link_counts.items(), key=lambda x: x[1], reverse=True)
                for link, count in sorted_links[:10]:
                    st.write(f"• {link} ({count} references)")
    
    with tab5:
        st.subheader("Content Analytics")
        
        if data:
            # Content statistics
            total_words = sum(item.get('word_count', 0) for item in data)
            total_images = sum(len(item.get('image_urls', [])) for item in data)
            avg_words = total_words / len(data) if data else 0
            avg_images = total_images / len(data) if data else 0
            
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
            
            # Content distribution charts
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Word count distribution
            word_counts = [item.get('word_count', 0) for item in data]
            depths = [item.get('depth', 0) for item in data]
            
            fig_words = px.histogram(x=word_counts, title="Word Count Distribution", 
                                   labels={'x': 'Words per Page', 'y': 'Number of Pages'})
            st.plotly_chart(fig_words, use_container_width=True)
            
            # Words by depth
            depth_words = {}
            for item in data:
                depth = item.get('depth', 0)
                words = item.get('word_count', 0)
                if depth not in depth_words:
                    depth_words[depth] = []
                depth_words[depth].append(words)
            
            depth_avg_words = {depth: sum(words)/len(words) for depth, words in depth_words.items()}
            
            fig_depth = px.bar(x=list(depth_avg_words.keys()), y=list(depth_avg_words.values()),
                             title="Average Words by Depth Level",
                             labels={'x': 'Depth Level', 'y': 'Average Words'})
            st.plotly_chart(fig_depth, use_container_width=True)
            
            # Top content pages
            st.write("**Most Content-Rich Pages:**")
            content_sorted = sorted(data, key=lambda x: x.get('word_count', 0), reverse=True)
            for item in content_sorted[:5]:
                title = item.get('title', 'No title')
                words = item.get('word_count', 0)
                images = len(item.get('image_urls', []))
                st.write(f"• **{title}** - {words:,} words, {images} images")
        else:
            st.info("No data available for analytics")
    
    with tab3:
        st.subheader("Summary Table")
        
        # Create summary dataframe
        summary_data = []
        for i, item in enumerate(data):
            summary_data.append({
                "Page": i + 1,
                "Path": item.get('path', 'Unknown'),
                "Title": item.get('title', 'No title')[:50] + "..." if len(item.get('title', '')) > 50 else item.get('title', 'No title'),
                "Content Length": len(item.get('content', '')),
                "Images Count": len(item.get('image_urls', [])),
                "Has Content": "Yes" if item.get('content', '').strip() else "No"
            })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)
            
            # Download options
            st.subheader("Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON download
                json_data = json.dumps(data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download as JSON",
                    data=json_data,
                    file_name=f"url_analysis_{int(time.time())}.json",
                    mime="application/json"
                )
            
            with col2:
                # CSV download
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download Summary as CSV",
                    data=csv_data,
                    file_name=f"url_analysis_summary_{int(time.time())}.csv",
                    mime="text/csv"
                )

def main():
    st.title("🔍 URL Analyzer")
    st.write("Enter a URL to analyze its content and extract images")
    
    # Check backend health
    if not check_backend_health():
        st.error("⚠️ Backend service is not available. Please ensure the backend is running on http://localhost:8002")
        st.stop()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("🔧 Analysis Settings")
        
        # Deep crawling settings
        st.subheader("Deep Crawling")
        follow_links = st.checkbox("Follow Links (Deep Crawl)", value=True, help="Enable deep crawling to analyze multiple pages")
        
        if follow_links:
            max_pages = st.slider("Max Pages", min_value=1, max_value=50, value=10, help="Maximum number of pages to analyze")
            max_depth = st.slider("Max Depth", min_value=1, max_value=5, value=2, help="Maximum crawling depth")
        else:
            max_pages = 1
            max_depth = 0
        
        st.divider()
        
        # Advanced options
        use_advanced = st.checkbox("Advanced Options")
        
        match_patterns = []
        content_selector = None
        
        if use_advanced:
            st.subheader("URL Pattern Matching")
            pattern_input = st.text_area(
                "Match Patterns (one per line)",
                help="Enter URL patterns to match (e.g., /products/, /blog/)",
                placeholder="/products/\n/services/"
            )
            if pattern_input:
                match_patterns = [p.strip() for p in pattern_input.split('\n') if p.strip()]
            
            st.subheader("Content Selector")
            content_selector = st.text_input(
                "CSS Selector",
                help="CSS selector for content extraction (e.g., .content, #main)",
                placeholder=".content"
            )
            if not content_selector:
                content_selector = None
        
        st.divider()
        
        # Show current settings
        st.subheader("📋 Current Settings")
        st.write(f"**Deep Crawl:** {'✅ Enabled' if follow_links else '❌ Single Page'}")
        if follow_links:
            st.write(f"**Max Pages:** {max_pages}")
            st.write(f"**Max Depth:** {max_depth}")
        st.write(f"**Match Patterns:** {len(match_patterns)} patterns" if match_patterns else "**Match Patterns:** All pages")
        st.write(f"**Content Selector:** {content_selector}" if content_selector else "**Content Selector:** Default")
    
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
        
        # Start analysis
        with st.spinner("Starting analysis..."):
            result = start_analysis(
                url_input, 
                max_pages=max_pages,
                max_depth=max_depth,
                follow_links=follow_links,
                match_patterns=match_patterns, 
                content_selector=content_selector
            )
        
        if result and "task_id" in result:
            task_id = result["task_id"]
            st.success(f"Analysis started! Task ID: {task_id}")
            
            # Progress tracking with enhanced display
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            # Add a container for detailed progress
            progress_container = st.container()
            
            # Poll for status with longer timeout for deep crawling
            max_wait_time = 600  # 10 minutes for deep crawling
            start_time = time.time()
            poll_interval = 1  # Poll every 1 second for better responsiveness
            
            st.info(f"⚙️ Analysis Settings: Max Pages: {max_pages}, Max Depth: {max_depth}, Follow Links: {follow_links}")
            
            while time.time() - start_time < max_wait_time:
                status_data = get_task_status(task_id)
                
                if not status_data:
                    st.error("Failed to get task status")
                    break
                
                # Use the enhanced progress display
                with progress_container:
                    is_complete = display_progress(status_data, progress_bar, status_placeholder)
                
                if is_complete:
                    result_data = status_data.get("result")
                    
                    if result_data and status_data.get("status") == "completed":
                        # Save results to session state
                        st.session_state.analysis_results = result_data
                        st.success("🎉 Analysis completed successfully!")
                        display_analysis_results(result_data)
                    elif status_data.get("status") == "failed":
                        error_msg = status_data.get("result", {}).get("error", "Unknown error")
                        st.error(f"❌ Analysis failed: {error_msg}")
                    else:
                        st.warning("Analysis completed but no results available")
                    break
                
                time.sleep(poll_interval)
            else:
                st.error("⏱️ Analysis timed out after 10 minutes. The process may still be running in the background.")
    
    # Display previous results if available
    if hasattr(st.session_state, 'analysis_results') and st.session_state.analysis_results:
        st.divider()
        st.subheader("📋 Previous Analysis Results")
        st.info("Showing results from previous analysis. You can still download the data below.")
        display_analysis_results(st.session_state.analysis_results)
    
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