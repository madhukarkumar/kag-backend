# SingleStore Knowledge Graph Backend API

This is the backend API server for the SingleStore Knowledge Graph application. It can be deployed both locally and on Replit or Railway.

## Local Development

### Prerequisites
- Python 3.10
- SingleStore database access
- OpenAI API key (for text processing and video transcription)
- Google Gemini API key
- FFmpeg (for video processing)
- Additional system libraries: libsm6, libxext6, libavcodec-extra

### Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install system dependencies (Ubuntu/Debian):
```bash
sudo apt-get update && sudo apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libavcodec-extra
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with:
```env
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
SINGLESTORE_HOST=your_host
SINGLESTORE_PORT=your_port
SINGLESTORE_USER=your_user
SINGLESTORE_PASSWORD=your_password
SINGLESTORE_DATABASE=your_database
```

5. Run the server locally:
```bash
./start_backend_services.sh  # On Windows: python main.py
```

The server will start at `http://localhost:8000`

## Features

### Document Processing
- PDF document processing with text extraction and analysis
- Video processing with audio transcription using OpenAI Whisper
- Automatic knowledge extraction from both PDFs and video transcripts
- Real-time processing status updates
- Background task processing with Celery

### Supported Formats
- PDF files
- Video files (.mp4, .mov, .avi, .mkv)

## API Endpoints

- `/upload-pdf`: Upload and process PDF documents
- `/upload-video`: Upload and process video files
- `/kbdata`: Knowledge base statistics
- `/config`: System configuration
- `/search`: Document search
- `/graph`: Knowledge graph
- `/task-status`: Processing status
- `/cancel-processing`: Cancel tasks

## API Documentation

Visit `/docs` or `/redoc` for complete API documentation.

## Environment Variables

See `.env.example` for all required environment variables.

## Docker Deployment

The application includes a Dockerfile configured with all necessary dependencies for both PDF and video processing. The Docker image:
- Uses Python 3.10
- Includes FFmpeg and required system libraries
- Sets up proper permissions and non-root user
- Configures Celery for background processing

## Notes

- Free tier Replit has limitations (512MB RAM, 500MB storage)
- Server sleeps after inactivity on free tier
- Consider "Always On" feature for production use
- Video processing requires significant CPU and memory resources
- Large video files may take longer to process due to transcription
