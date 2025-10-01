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

## Architecture & Workflow

### Core Components
- **Frontend**: Streamlit web application
- **AI Processing**: Claude Sonnet 4 via Vertex AI
- **OCR Fallback**: AWS Textract for table extraction
- **Storage**: AWS S3 for file storage
- **Metadata**: Google Sheets for tracking
- **Caching**: Local CSV cache for analysis results

### Application Workflow

1. **Initial Setup & Configuration**
   - Environment validation (`check_env_file()`)
   - AWS credentials setup (`setup_aws_credentials()`)
   - Google Cloud credentials setup (`setup_google_credentials()`)
   - Client initialization (`setup_clients()`) for S3, Textract, and Google services

2. **Statement Type Management**
   - Pre-defined statement aliases (`setup_statement_aliases()`)
   - Supports Balance Sheet, Cash Flow, and Income Statements
   - Prioritizes consolidated/group statements over standard statements

3. **PDF Analysis & Statement Location**
   - Analyzes PDFs to locate specific financial statements (`analyze_all_statements_in_pdf()`)
   - Smart page detection with multi-page support (`find_statement_page_robust()`)
   - Caching system to store analysis results (`setup_metadata_cache()`)

4. **Data Extraction Methods**
   - Claude AI extraction (`extract_with_claude()`)
     - Handles multi-page statements
     - Maintains hierarchical structure
     - Consolidates data into standardized CSV format
   
   - AWS Textract extraction (`extract_with_textract()`)
     - Table structure recognition
     - Multi-page support
     - CSV conversion

5. **Results Processing & Storage**
   - CSV data generation
   - S3 upload functionality (`upload_to_s3()`)
   - Google Sheets integration (`add_row_to_spreadsheet()`)

### Key Functions Overview

```python
# Setup and Configuration
FinancialValidationApp
├── setup_aws_credentials()      # AWS authentication
├── setup_google_credentials()   # Google Cloud auth
├── setup_clients()             # Initialize service clients
└── setup_statement_aliases()    # Define statement types

# Caching System
├── setup_metadata_cache()       # Initialize cache system
├── check_metadata_cache()       # Check for existing results
├── save_to_metadata_cache()     # Store analysis results
└── clear_metadata_cache()       # Reset cache

# PDF Analysis
├── find_statement_page_robust()           # Locate statements
├── analyze_all_statements_in_pdf()        # Full PDF analysis
├── extract_pdf_chunk()                    # Extract specific pages
└── find_statement_page_robust_with_cache() # Cached analysis

# Data Extraction
├── extract_with_claude()                  # AI-based extraction
├── extract_with_textract()                # AWS Textract extraction
├── consolidate_tables_to_csv()            # Table consolidation
└── extract_tables_from_blocks()           # Textract parsing

# Storage & Upload
├── upload_to_s3()              # S3 upload with debugging
├── upload_csv_to_s3()          # CSV-specific upload
└── add_row_to_spreadsheet()    # Google Sheets integration
```

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

## Best Practices for Integration

When integrating these components into a new project:

1. **Core Setup**
   - Start with the `FinancialValidationApp` class
   - Implement essential environment variables
   - Set up AWS and Google Cloud credentials

2. **Caching Implementation**
   - Use the caching system for performance optimization
   - Implement the cache file structure with required columns
   - Maintain cache validation and cleanup

3. **Extraction Method Selection**
   - Choose between Claude AI and Textract based on needs
   - Consider implementing both for fallback support
   - Test accuracy with your specific document types

4. **Storage Integration**
   - Set up S3 buckets with appropriate permissions
   - Implement the upload functions with error handling
   - Add Google Sheets integration if metadata tracking is needed

5. **Error Handling**
   - Maintain comprehensive try-catch blocks
   - Implement logging and debugging
   - Add validation at each processing step

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
