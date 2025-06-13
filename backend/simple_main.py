import asyncio
import os
import time
import sys
from typing import Optional, List, Dict, Set
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import traceback
import json
import re
from urllib.parse import urljoin, urlparse
import aiohttp
from bs4 import BeautifulSoup

app = FastAPI(title="URL Analyzer API", description="Standalone URL analysis system")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class URLAnalysisRequest(BaseModel):
    url: HttpUrl
    max_pages: Optional[int] = 10
    max_depth: Optional[int] = 2
    match_patterns: Optional[List[str]] = None
    content_selector: Optional[str] = None
    follow_links: Optional[bool] = True

class URLAnalysisResult(BaseModel):
    url: str
    status: str
    total_pages: int
    total_images: int
    analysis_time: float
    data: List[dict]
    error: Optional[str] = None

class TaskStatus(BaseModel):
    task_id: str
    status: str
    message: str
    progress: Optional[Dict] = None
    result: Optional[URLAnalysisResult] = None

# Global storage for task status
task_status: dict[str, TaskStatus] = {}

def generate_task_id(url: str) -> str:
    """Generate a unique task ID based on URL and timestamp"""
    import hashlib
    return hashlib.md5(f"{url}_{int(time.time())}".encode()).hexdigest()[:12]

def update_task_progress(task_id: str, status: str, message: str, progress: Optional[Dict] = None):
    """Update task progress"""
    if task_id in task_status:
        task_status[task_id].status = status
        task_status[task_id].message = message
        if progress:
            task_status[task_id].progress = progress
        print(f"Progress Update [{task_id}]: {status} - {message}")
        if progress:
            print(f"Progress Details: {progress}")

class DeepCrawler:
    """Deep crawling functionality for comprehensive site analysis"""
    
    def __init__(self, max_concurrent: int = 5, max_pages: int = 10, max_depth: int = 2, task_id: str = None):
        self.max_concurrent = max_concurrent
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.task_id = task_id
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.visited_urls: Set[str] = set()
        self.found_pages: List[Dict] = []
        self.crawl_queue: List[str] = []
        self.start_time = time.time()
    
    def report_progress(self, message: str, status: str = "processing"):
        """Report progress to task status"""
        if self.task_id:
            elapsed = time.time() - self.start_time
            progress = {
                "pages_found": len(self.found_pages),
                "pages_visited": len(self.visited_urls),
                "max_pages": self.max_pages,
                "max_depth": self.max_depth,
                "elapsed_time": round(elapsed, 2),
                "queue_size": len(self.crawl_queue)
            }
            update_task_progress(self.task_id, status, message, progress)
        
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
                    for link in new_links[:5]:  # Limit concurrent crawls
                        if link not in self.visited_urls:
                            crawl_tasks.append(self.crawl_page(link, depth + 1))
                    
                    if crawl_tasks:
                        self.report_progress(f"Starting crawl of {len(crawl_tasks)} linked pages from {url}")
                        await asyncio.gather(*crawl_tasks, return_exceptions=True)
                
                return page_data
                
            except Exception as e:
                print(f"Error crawling {url}: {e}")
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
            bg_matches = re.findall(r'background-image:\s*url\(["\']?([^"\')\s]+)["\']?\)', style)
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
        self.report_progress(f"Site crawl completed - {len(self.found_pages)} pages found", "completed")
        return self.found_pages

async def extract_basic_content(url: str) -> dict:
    """Extract basic content from a single URL using simple HTTP request"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=30) as response:
                if response.status != 200:
                    return {
                        "path": "/",
                        "title": "Error",
                        "content": f"HTTP {response.status}: Could not fetch content",
                        "image_urls": []
                    }
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract title
                title_tag = soup.find('title')
                title = title_tag.get_text().strip() if title_tag else "No title"
                
                # Extract text content
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                text_content = soup.get_text()
                lines = (line.strip() for line in text_content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                content = ' '.join(chunk for chunk in chunks if chunk)
                
                # Extract image URLs
                image_urls = []
                for img in soup.find_all('img'):
                    src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                    if src:
                        # Convert relative URLs to absolute
                        absolute_url = urljoin(url, src)
                        image_urls.append(absolute_url)
                
                # Remove duplicates
                image_urls = list(set(image_urls))
                
                return {
                    "path": "/",
                    "title": title,
                    "content": content[:5000],  # Limit content length
                    "image_urls": image_urls
                }
                
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return {
            "path": "/",
            "title": "Error",
            "content": f"Error extracting content: {str(e)}",
            "image_urls": []
        }

async def analyze_url_task(task_id: str, url: str, max_pages: int, max_depth: int, follow_links: bool, match_patterns: Optional[List[str]], content_selector: Optional[str]):
    """Background task to analyze URL using deep crawling"""
    try:
        # Update status to processing
        task_status[task_id].status = "processing"
        task_status[task_id].message = "Starting deep analysis..."
        
        print(f"Starting deep analysis for URL: {url} (max_pages: {max_pages}, max_depth: {max_depth})")
        start_time = time.time()
        
        if follow_links and max_depth > 0:
            # Use deep crawler for comprehensive analysis
            crawler = DeepCrawler(
                max_concurrent=5, 
                max_pages=max_pages, 
                max_depth=max_depth,
                task_id=task_id  # Pass task_id for progress reporting
            )
            
            update_task_progress(task_id, "processing", f"Initializing crawler (max {max_pages} pages, depth {max_depth})")
            collected_data = await crawler.crawl_site(url)
        else:
            # Single page analysis
            update_task_progress(task_id, "processing", f"Analyzing single page: {url}")
            content_data = await extract_basic_content(url)
            collected_data = [content_data] if content_data else []
            if collected_data:
                update_task_progress(task_id, "processing", f"Single page analysis completed")
        
        end_time = time.time()
        analysis_time = end_time - start_time
        
        if not collected_data:
            task_status[task_id].status = "completed"
            task_status[task_id].message = "Analysis completed - no content found"
            task_status[task_id].result = URLAnalysisResult(
                url=url,
                status="no_content",
                total_pages=0,
                total_images=0,
                analysis_time=analysis_time,
                data=[]
            )
            return
        
        # Calculate statistics
        total_pages = len(collected_data)
        total_images = sum(len(item.get("image_urls", [])) for item in collected_data)
        total_words = sum(item.get("word_count", 0) for item in collected_data)
        
        # Update status to completed
        final_progress = {
            "pages_found": total_pages,
            "total_images": total_images,
            "total_words": total_words,
            "analysis_time": analysis_time,
            "status": "completed"
        }
        
        update_task_progress(
            task_id, 
            "completed", 
            f"Analysis completed - found {total_pages} pages with {total_images} images", 
            final_progress
        )
        
        task_status[task_id].result = URLAnalysisResult(
            url=url,
            status="success",
            total_pages=total_pages,
            total_images=total_images,
            analysis_time=analysis_time,
            data=collected_data
        )
        
        print(f"Deep analysis completed for {url}: {total_pages} pages, {total_images} images, {total_words} words in {analysis_time:.2f}s")
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error analyzing URL {url}: {error_msg}")
        traceback.print_exc()
        
        task_status[task_id].status = "failed"
        task_status[task_id].message = f"Analysis failed: {error_msg}"
        task_status[task_id].result = URLAnalysisResult(
            url=url,
            status="failed",
            total_pages=0,
            total_images=0,
            analysis_time=time.time() - start_time if 'start_time' in locals() else 0,
            data=[],
            error=error_msg
        )

@app.post("/analyze", response_model=dict)
async def start_url_analysis(request: URLAnalysisRequest, background_tasks: BackgroundTasks):
    """Start URL analysis in background"""
    url = str(request.url)
    task_id = generate_task_id(url)
    
    # Initialize task status
    task_status[task_id] = TaskStatus(
        task_id=task_id,
        status="queued",
        message="Analysis queued"
    )
    
    # Start background task with enhanced parameters
    background_tasks.add_task(
        analyze_url_task,
        task_id,
        url,
        request.max_pages or 10,
        request.max_depth or 2,
        request.follow_links if request.follow_links is not None else True,
        request.match_patterns,
        request.content_selector
    )
    
    return {
        "task_id": task_id,
        "message": "Deep URL analysis started",
        "status_url": f"/status/{task_id}",
        "settings": {
            "max_pages": request.max_pages or 10,
            "max_depth": request.max_depth or 2,
            "follow_links": request.follow_links if request.follow_links is not None else True
        }
    }

@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get status of analysis task"""
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_status[task_id]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "URL Analyzer API is running"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "URL Analyzer API (Simple)",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "POST /analyze - Start URL analysis",
            "status": "GET /status/{task_id} - Get task status",
            "health": "GET /health - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)