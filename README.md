# SingleStore Knowledge Graph Backend API

This is the backend API server for the SingleStore Knowledge Graph application. It can be deployed both locally and on Replit or Railway.

## Local Development

### Prerequisites
- Python 3.12.9
- SingleStore database access
- OpenAI API key
- Google Gemini API key

### Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with:
```env
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
SINGLESTORE_HOST=your_host
SINGLESTORE_PORT=your_port
SINGLESTORE_USER=your_user
SINGLESTORE_PASSWORD=your_password
SINGLESTORE_DATABASE=your_database
```

4. Run the server locally:
```bash
./start_backend_services.sh  # On Windows: python main.py
```

The server will start at `http://localhost:8000`

## Replit Deployment

1. Create a new Python repl on [Replit](https://replit.com)
2. Upload all the files from this directory
3. Add environment variables in Replit Secrets
4. Click "Run" to deploy

The server will be available at your Replit URL (e.g., `https://your-repl-name.username.repl.co`)

## Railway Deployment

1. Fork this repository
2. Create a new project on [Railway](https://railway.app)
3. Connect your GitHub repository
4. Add the following environment variables in Railway:
   ```env
   OPENAI_API_KEY=your_openai_key
   GEMINI_API_KEY=your_gemini_key
   SINGLESTORE_HOST=your_host
   SINGLESTORE_PORT=your_port
   SINGLESTORE_USER=your_user
   SINGLESTORE_PASSWORD=your_password
   SINGLESTORE_DATABASE=your_database
   API_KEY=your_secure_api_key
   ```
5. Deploy! Railway will automatically build and deploy your application

The server will be available at your Railway URL (e.g., `https://your-app-name.railway.app`)

## Project Structure
```
.
├── .replit                 # Replit configuration
├── api/                    # API endpoints
├── config/                 # Configuration management
├── core/                   # Core business logic
├── db/                     # Database operations
├── processors/             # Document processors
├── search/                 # Search functionality
├── tasks/                  # Background tasks
├── utils/                  # Utility functions
├── main.py                # FastAPI application
├── requirements.txt       # Python dependencies
└── start_backend_services.sh  # Local startup script
```

## API Endpoints

- `/kbdata`: Knowledge base statistics
- `/config`: System configuration
- `/upload`: Document upload
- `/search`: Document search
- `/graph`: Knowledge graph
- `/task-status`: Processing status
- `/cancel-processing`: Cancel tasks

## API Documentation

Visit `/docs` or `/redoc` for complete API documentation.

## Environment Variables

See `.env.example` for all required environment variables.

## Notes

- Free tier Replit has limitations (512MB RAM, 500MB storage)
- Server sleeps after inactivity on free tier
- Consider "Always On" feature for production use
