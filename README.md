# Financial Statement Extraction & Upload Tool

A Streamlit application for extracting financial statements from PDF documents using AI (Claude Sonnet 4) and AWS Textract, with automated upload to S3 and Google Sheets integration.

## Features

- PDF financial statement detection and extraction
- Multi-page statement support
- AI-powered page location with caching
- Consolidated vs standard statement prioritization
- AWS S3 integration for file storage
- Google Sheets metadata tracking
- Docker containerization for easy deployment

## Prerequisites

- Docker and Docker Compose
- AWS Account with S3 and Textract access
- Google Cloud Project with Vertex AI enabled
- Google Service Account with appropriate permissions

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd financial-extractor-docker
```

### 2. Environment Configuration

```bash
# Copy the environment template
cp .env.template .env

# Edit .env with your actual credentials
nano .env
```

Required environment variables:
- `AWS_ACCESS_KEY_ID` - Your AWS access key
- `AWS_SECRET_ACCESS_KEY` - Your AWS secret key
- `AWS_REGION` - AWS region (default: us-east-1)
- `PROJECT_ID` - Google Cloud project ID
- `VERTEX_REGION` - Vertex AI region (default: us-east5)
- `SERVICE_ACCOUNT_JSON` - Google service account JSON (full content)
- `TEMP_BUCKET` - S3 bucket for temporary files

### 3. Build and Run

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

### 4. Access the Application

Open your browser and navigate to:
- `http://localhost:8501` (or `http://localhost:8502` if using alternate port)

## Usage

1. **Load CSV**: The application automatically loads financial statements metadata
2. **Filter Records**: Select symbol, statement type, period, and report type
3. **Generate Content Overview**: Analyze PDF for all available statements
4. **Extract Data**: Choose individual statements to extract
5. **Upload to S3**: Automatically upload extracted CSV files
6. **Update Spreadsheet**: Sync metadata with Google Sheets

## Architecture

- **Frontend**: Streamlit web application
- **AI Processing**: Claude Sonnet 4 via Vertex AI
- **OCR Fallback**: AWS Textract for table extraction
- **Storage**: AWS S3 for file storage
- **Metadata**: Google Sheets for tracking
- **Caching**: Local CSV cache for analysis results

## Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
# ... other variables

# Run locally
streamlit run app.py
```

### Docker Development

```bash
# Build image
docker build -t financial-extractor .

# Run with environment file
docker run --env-file .env -p 8501:8501 financial-extractor
```

## Troubleshooting

### Port Already in Use
```bash
# Check what's using port 8501
lsof -i :8501

# Use different port
docker-compose up --build -p 8502:8501
```

### Cache Issues
```bash
# Clear cache from the web interface
# Or manually delete pdf_analysis_cache.csv
```

### AWS Connection Issues
- Verify AWS credentials in .env file
- Check S3 bucket permissions
- Test AWS connection using the built-in test feature

### Google Cloud Issues
- Verify service account JSON is properly formatted
- Check Vertex AI API is enabled
- Verify project ID and region settings

## Security Notes

- Never commit `.env` files to version control
- Use Docker secrets for production deployments
- Regularly rotate AWS and Google Cloud credentials
- Review S3 bucket permissions periodically

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Add your license information here]

## Support

[Add support contact information]
