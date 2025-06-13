# URL Analyzer

A comprehensive web URL analysis system with deep crawling capabilities, built with FastAPI backend and Streamlit frontend.

## Features

- **Deep Web Crawling**: Analyze multiple pages with configurable depth and page limits
- **Comprehensive Image Extraction**: Extract images from img tags, CSS backgrounds, picture elements, and lazy-loading attributes
- **Real-time Progress Tracking**: Monitor analysis progress with detailed metrics and status updates
- **Interactive Web UI**: User-friendly Streamlit interface with advanced filtering and visualization
- **Parallel Processing**: Efficient concurrent page analysis with semaphore-controlled requests
- **Data Export**: Download results in JSON or CSV format
- **Docker Support**: Easy deployment with Docker Compose

## Architecture

- **Backend**: FastAPI with async/await for high-performance web crawling
- **Frontend**: Streamlit for interactive data visualization and analysis
- **Containerization**: Docker Compose for easy deployment and scaling
- **Data Processing**: BeautifulSoup for HTML parsing, aiohttp for async HTTP requests

## Quick Start

### Using Docker (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd url_analyzer
```

2. Start the services:
```bash
docker-compose up --build
```

3. Access the application:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8002

### Manual Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the backend:
```bash
python backend/simple_main.py
```

3. Start the frontend (in another terminal):
```bash
streamlit run frontend/app.py
```

## Usage

### Web Interface

1. Open http://localhost:8501 in your browser
2. Enter a URL to analyze
3. Configure analysis settings:
   - **Deep Crawl**: Enable/disable multi-page analysis
   - **Max Pages**: Maximum number of pages to crawl (1-50)
   - **Max Depth**: Maximum crawling depth (1-5)
   - **Match Patterns**: URL patterns to include
   - **Content Selector**: CSS selector for specific content

4. Click "Start Analysis" and monitor real-time progress
5. View results in multiple tabs:
   - **Page Content**: Detailed page analysis with filtering
   - **Images**: Image gallery organized by page
   - **Summary Table**: Tabular data overview
   - **Site Structure**: Hierarchical site map and link analysis
   - **Analytics**: Content statistics and visualizations

6. Download results as JSON or CSV

### API Usage

#### Start Analysis
```bash
POST /analyze
Content-Type: application/json

{
  "url": "https://example.com",
  "max_pages": 10,
  "max_depth": 2,
  "follow_links": true,
  "match_patterns": ["/products/"],
  "content_selector": ".content"
}
```

#### Check Status
```bash
GET /status/{task_id}
```

#### Health Check
```bash
GET /health
```

## API Endpoints

- `POST /analyze` - Start URL analysis
- `GET /status/{task_id}` - Get analysis status
- `GET /health` - Health check
- `GET /` - API information

## Configuration

### Environment Variables

- `AZURE_OPENAI_API_KEY` - Required for browser automation
- `AZURE_OPENAI_ENDPOINT` - Azure OpenAI endpoint
- `AZURE_OPENAI_DEPLOYMENT_NAME` - Model deployment name
- `BROWSER_HEADLESS` - Run browser in headless mode (default: true)
- `MCP_MODEL_PROVIDER` - Model provider (default: azure_openai)

### Advanced Options

- **Match Patterns**: Filter URLs by patterns (e.g., `/products/`, `/blog/`)
- **Content Selector**: CSS selector for content extraction
- **Concurrency**: Adjust concurrent processing in the content collector

## Output Format

The analysis returns:
```json
{
  "url": "https://example.com",
  "status": "success",
  "total_pages": 10,
  "total_images": 25,
  "analysis_time": 45.2,
  "data": [
    {
      "path": "/page1",
      "title": "Page Title",
      "content": "Page content...",
      "image_urls": ["https://example.com/image1.jpg"]
    }
  ]
}
```

## Integration with Existing System

This URL Analyzer is designed to work alongside the main sitemcp-chatbot system:

- **Shared Dependencies**: Uses `mcp_use` and `mcp-browser-use` packages
- **Content Collection**: Leverages `content_collector.py` functionality
- **Environment**: Shares environment configuration

## Troubleshooting

### Common Issues

1. **Backend Not Available**: Ensure the backend is running on port 8002
2. **Missing Dependencies**: Install required packages with `pip install -r requirements.txt`
3. **Environment Variables**: Check that Azure OpenAI credentials are properly set
4. **Docker Issues**: Ensure Docker and Docker Compose are installed

### Logs

- Backend logs: Check console output from the FastAPI service
- Frontend logs: Check browser console and Streamlit terminal output
- Docker logs: `docker-compose logs [service-name]`

## Development

### Project Structure
```
url_analyzer/
├── backend/
│   └── main.py          # FastAPI backend
├── frontend/
│   └── app.py           # Streamlit frontend
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container definition
├── docker-compose.yml  # Service orchestration
└── README.md           # This file
```

### Adding Features

1. **Backend**: Modify `backend/main.py` to add new API endpoints
2. **Frontend**: Update `frontend/app.py` to add UI components
3. **Dependencies**: Add new packages to `requirements.txt`

## License

This project inherits the license from the parent sitemcp-chatbot project.