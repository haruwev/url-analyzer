import asyncio
import os
import time
from typing import Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import sys
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import existing content collection system
from content_collector import collect_site_content

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
    match_patterns: Optional[List[str]] = None
    content_selector: Optional[str] = None

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
    result: Optional[URLAnalysisResult] = None

# Global storage for task status
task_status: dict[str, TaskStatus] = {}

def generate_task_id(url: str) -> str:
    """Generate a unique task ID based on URL and timestamp"""
    import hashlib
    return hashlib.md5(f"{url}_{int(time.time())}".encode()).hexdigest()[:12]

async def analyze_url_task(task_id: str, url: str, match_patterns: Optional[List[str]], content_selector: Optional[str]):
    """Background task to analyze URL"""
    try:
        # Update status to processing
        task_status[task_id].status = "processing"
        task_status[task_id].message = "Analyzing URL content..."
        
        print(f"Starting analysis for URL: {url}")
        start_time = time.time()
        
        # Use existing content collection system
        collected_data = await collect_site_content(
            url=url,
            match_patterns=match_patterns,
            content_selector=content_selector
        )
        
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
        
        # Update status to completed
        task_status[task_id].status = "completed"
        task_status[task_id].message = f"Analysis completed - found {total_pages} pages with {total_images} images"
        task_status[task_id].result = URLAnalysisResult(
            url=url,
            status="success",
            total_pages=total_pages,
            total_images=total_images,
            analysis_time=analysis_time,
            data=collected_data
        )
        
        print(f"Analysis completed for {url}: {total_pages} pages, {total_images} images in {analysis_time:.2f}s")
        
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
    
    # Start background task
    background_tasks.add_task(
        analyze_url_task,
        task_id,
        url,
        request.match_patterns,
        request.content_selector
    )
    
    return {
        "task_id": task_id,
        "message": "URL analysis started",
        "status_url": f"/status/{task_id}"
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
        "service": "URL Analyzer API",
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