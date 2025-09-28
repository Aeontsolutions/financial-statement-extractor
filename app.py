import streamlit as st
import pandas as pd
import boto3
import json
import base64
import requests
import io
import glob
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import fitz  # PyMuPDF for PDF rendering
from urllib.parse import urlparse
from botocore.exceptions import ClientError
import time

# Load environment variables
load_dotenv()

class FinancialValidationApp:
    def __init__(self):
        self.setup_aws_credentials()
        self.setup_google_credentials()
        self.setup_clients()
        self.setup_statement_aliases()
        self.setup_metadata_cache()
        
    def setup_metadata_cache(self):
        """Setup metadata caching system."""
        self.cache_file = "pdf_analysis_cache.csv"
        self.cache_columns = [
            'pdf_s3_path', 'symbol', 'period_end_date', 'report_type', 'statement_type',
            'pdf_page_positions', 'printed_page_numbers', 'statement_title', 
            'statement_priority', 'is_multi_page', 'confidence', 'reasoning',
            'analysis_timestamp', 'pdf_hash'
        ]
        self.initialize_cache_file()
    
    def initialize_cache_file(self):
        """Initialize the cache file if it doesn't exist."""
        try:
            if not os.path.exists(self.cache_file):
                # Create empty cache file with headers
                cache_df = pd.DataFrame(columns=self.cache_columns)
                cache_df.to_csv(self.cache_file, index=False)
                st.info(f"Initialized metadata cache: {self.cache_file}")
        except Exception as e:
            st.warning(f"Could not initialize cache file: {e}")
    
    def generate_pdf_hash(self, pdf_content: bytes) -> str:
        """Generate a hash of the PDF content for cache validation."""
        try:
            import hashlib
            return hashlib.md5(pdf_content).hexdigest()[:16]  # Use first 16 chars for brevity
        except Exception as e:
            st.warning(f"Could not generate PDF hash: {e}")
            return "unknown"
    
    def get_cache_key(self, pdf_s3_path: str, symbol: str, period_end_date: str, report_type: str, statement_type: str) -> str:
        """Generate a unique cache key for the analysis."""
        return f"{symbol}_{period_end_date}_{report_type}_{statement_type}_{pdf_s3_path}"
    
    def check_metadata_cache(self, pdf_s3_path: str, symbol: str, period_end_date: str, report_type: str, statement_type: str, pdf_content: bytes) -> Optional[Dict]:
        """Check if analysis results exist in cache."""
        try:
            if not os.path.exists(self.cache_file):
                st.info("No cache file exists yet")
                return None
            
            # Load cache
            cache_df = pd.read_csv(self.cache_file)
            
            if cache_df.empty:
                st.info("Cache file is empty")
                return None
            
            # Debug: Show what we're looking for
            st.info(f"Checking cache for: {symbol}_{statement_type}_{period_end_date}")
            st.info(f"Cache contains {len(cache_df)} entries")
            
            # Generate current PDF hash for validation
            current_pdf_hash = self.generate_pdf_hash(pdf_content)
            
            # Look for matching entry
            mask = (
                (cache_df['pdf_s3_path'] == pdf_s3_path) &
                (cache_df['symbol'] == symbol) &
                (cache_df['period_end_date'] == str(period_end_date)) &
                (cache_df['report_type'] == report_type) &
                (cache_df['statement_type'] == statement_type)
            )
            
            matching_entries = cache_df[mask]
            
            # Debug: Show matching results
            st.info(f"Found {len(matching_entries)} matching entries")
            
            if not matching_entries.empty:
                latest_entry = matching_entries.iloc[-1]  # Get most recent entry
                
                # Validate PDF hash if available
                cached_hash = latest_entry.get('pdf_hash', 'unknown')
                if cached_hash != 'unknown' and cached_hash != current_pdf_hash:
                    st.info("PDF content has changed since cache entry. Will re-analyze.")
                    return None
                
                # Parse the cached data
                try:
                    pdf_page_positions = eval(latest_entry['pdf_page_positions']) if pd.notna(latest_entry['pdf_page_positions']) else []
                    printed_page_numbers = eval(latest_entry['printed_page_numbers']) if pd.notna(latest_entry['printed_page_numbers']) else []
                except:
                    pdf_page_positions = []
                    printed_page_numbers = []
                
                cached_result = {
                    'pdf_page_positions': pdf_page_positions,
                    'printed_page_numbers': printed_page_numbers,
                    'statement_title': latest_entry.get('statement_title', 'Unknown'),
                    'statement_priority': latest_entry.get('statement_priority', 'standard'),
                    'is_multi_page': latest_entry.get('is_multi_page', False),
                    'confidence': latest_entry.get('confidence', 'medium'),
                    'reasoning': latest_entry.get('reasoning', 'Retrieved from cache'),
                    'from_cache': True,
                    'cache_timestamp': latest_entry.get('analysis_timestamp', 'Unknown')
                }
                
                st.success(f"Found cached analysis result (from {latest_entry.get('analysis_timestamp', 'unknown time')})")
                return cached_result
            else:
                st.info("No matching cache entries found")
            
            return None
            
        except Exception as e:
            st.error(f"Error checking metadata cache: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def save_to_metadata_cache(self, pdf_s3_path: str, symbol: str, period_end_date: str, report_type: str, statement_type: str, analysis_result: Dict, pdf_content: bytes):
        """Save analysis results to cache."""
        try:
            # Generate PDF hash
            pdf_hash = self.generate_pdf_hash(pdf_content)
            
            # Debug: Show what we're trying to cache
            st.info(f"Saving to cache: {symbol}_{statement_type}_{period_end_date}")
            
            # Prepare cache entry
            cache_entry = {
                'pdf_s3_path': pdf_s3_path,
                'symbol': symbol,
                'period_end_date': str(period_end_date),
                'report_type': report_type,
                'statement_type': statement_type,
                'pdf_page_positions': str(analysis_result.get('pdf_page_positions', [])),
                'printed_page_numbers': str(analysis_result.get('printed_page_numbers', [])),
                'statement_title': analysis_result.get('statement_title', 'Unknown'),
                'statement_priority': analysis_result.get('statement_priority', 'standard'),
                'is_multi_page': analysis_result.get('is_multi_page', False),
                'confidence': analysis_result.get('confidence', 'medium'),
                'reasoning': analysis_result.get('reasoning', ''),
                'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'pdf_hash': pdf_hash
            }
            
            # Load existing cache or create new
            if os.path.exists(self.cache_file):
                cache_df = pd.read_csv(self.cache_file)
            else:
                cache_df = pd.DataFrame(columns=self.cache_columns)
            
            # Add new entry
            new_entry_df = pd.DataFrame([cache_entry])
            cache_df = pd.concat([cache_df, new_entry_df], ignore_index=True)
            
            # Save to file
            cache_df.to_csv(self.cache_file, index=False)
            st.success(f"Saved analysis result to cache: {self.cache_file}")
            
            # Debug: Show cache file size
            st.info(f"Cache now contains {len(cache_df)} entries")
            
        except Exception as e:
            st.error(f"Could not save to metadata cache: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
    
    def clear_metadata_cache(self):
        """Clear the metadata cache file."""
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
                self.initialize_cache_file()
                st.success("Metadata cache cleared successfully")
            else:
                st.info("No cache file to clear")
        except Exception as e:
            st.error(f"Error clearing cache: {e}")
    
    def show_cache_stats(self):
        """Display cache statistics."""
        try:
            st.subheader("Cache Statistics")
            
            # Show file path being used
            st.write(f"**Cache file path:** `{os.path.abspath(self.cache_file)}`")
            st.write(f"**File exists:** {os.path.exists(self.cache_file)}")
            
            if not os.path.exists(self.cache_file):
                st.info("No cache file exists")
                return
            
            # Show file size
            file_size = os.path.getsize(self.cache_file)
            st.write(f"**File size:** {file_size} bytes")
            
            # Try to read the file
            try:
                cache_df = pd.read_csv(self.cache_file)
                
                if cache_df.empty:
                    st.warning("Cache file exists but is empty")
                    # Show raw file content for debugging
                    with open(self.cache_file, 'r') as f:
                        content = f.read()
                        st.text_area("Raw file content:", content, height=100)
                    return
                
                st.success(f"**Total cached entries:** {len(cache_df)}")
                st.write(f"**Unique symbols:** {cache_df['symbol'].nunique()}")
                
                # Show column names
                st.write(f"**Columns:** {list(cache_df.columns)}")
                
                # Show recent entries
                st.write("**All cache entries:**")
                st.dataframe(cache_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error reading cache file: {e}")
                # Show raw file content for debugging
                try:
                    with open(self.cache_file, 'r') as f:
                        content = f.read()
                        st.text_area("Raw file content (first 500 chars):", content[:500], height=100)
                except Exception as read_error:
                    st.error(f"Cannot even read raw file: {read_error}")
                
        except Exception as e:
            st.error(f"Error showing cache stats: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")

    def find_statement_page_robust(self, pdf_content: bytes, statement_type: str, report_type: str) -> Dict:
        """
        Enhanced page finding method with better handling of multi-page statements.
        This is the missing method that was causing the error.
        """
        try:
            pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
            
            # Create comprehensive prompt for finding statements
            consolidated_terms, standard_terms = self.get_statement_search_terms(statement_type)
            all_terms = consolidated_terms + standard_terms
            
            prompt = f"""
I need you to find a specific financial statement in this PDF with enhanced analysis.

TARGET: {statement_type}
REPORT TYPE: {report_type}

PRIORITY SEARCH TERMS (in order of preference):
1. HIGHEST PRIORITY - Consolidated/Group statements:
"""
            
            for term in consolidated_terms:
                prompt += f"   - {term}\n"
            
            if standard_terms:
                prompt += "\n2. FALLBACK - Standard statements:\n"
                for term in standard_terms:
                    prompt += f"   - {term}\n"
            
            prompt += f"""

ENHANCED ANALYSIS INSTRUCTIONS:

STEP 1: Create a comprehensive page inventory
Go through the PDF and catalog each page:
- Page 1: [brief content description]
- Page 2: [brief content description]
etc.

STEP 2: Multi-page statement detection
Look for statements that may span multiple pages. Consider:
- Continuation headers (e.g., "Continued", "Page 2 of 3")
- Partial tables that continue on next page
- "Carried forward" or "Brought forward" references
- Consistent formatting across pages

STEP 3: Prioritized search with multi-page support
Search for the target statement following priority:
1. FIRST: Consolidated/group versions (highest priority)
2. FALLBACK: Standard versions (only if consolidated not found)

STEP 4: Result format
Return results in this JSON format:

For SINGLE PAGE statements:
{{
  "pdf_page_positions": [page_number],
  "printed_page_numbers": ["printed_page_if_visible"],
  "statement_title": "exact_title_as_appears",
  "statement_priority": "consolidated/standard",
  "is_multi_page": false,
  "confidence": "high/medium/low",
  "reasoning": "explanation_of_findings"
}}

For MULTI-PAGE statements:
{{
  "pdf_page_positions": [page1, page2, page3],
  "printed_page_numbers": ["printed1", "printed2", "printed3"],
  "statement_title": "exact_title_as_appears",
  "statement_priority": "consolidated/standard",
  "is_multi_page": true,
  "confidence": "high/medium/low",
  "reasoning": "explanation_of_multi_page_detection"
}}

IMPORTANT: Use PDF sequential page numbers (1st page = 1, 2nd page = 2, etc.), NOT printed page numbers.
"""
            
            access_token = self.get_access_token()
            
            payload = {
                "anthropic_version": "vertex-2023-10-16",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": 3000,
                "temperature": 0
            }
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                self.vertex_endpoint,
                headers=headers,
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                response_data = response.json()
                if 'content' in response_data and len(response_data['content']) > 0:
                    response_text = response_data['content'][0]['text']
                    
                    # Extract JSON from response
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        result = json.loads(json_str)
                        
                        # Ensure pdf_page_positions is always a list
                        if 'pdf_page_positions' in result:
                            if not isinstance(result['pdf_page_positions'], list):
                                result['pdf_page_positions'] = [result['pdf_page_positions']]
                        else:
                            # Fallback to single page format
                            if 'pdf_page_position' in result:
                                result['pdf_page_positions'] = [result['pdf_page_position']]
                        
                        return result
            
            return None
                
        except Exception as e:
            st.error(f"Error in robust page finding: {e}")
            return None

    def find_statement_page_robust_with_cache(self, pdf_content: bytes, statement_type: str, report_type: str, pdf_s3_path: str, symbol: str, period_end_date: str) -> Dict:
        """Enhanced page finding with caching support."""
        try:
            # Check cache first
            cached_result = self.check_metadata_cache(pdf_s3_path, symbol, period_end_date, report_type, statement_type, pdf_content)
            
            if cached_result:
                # Mark as from cache for UI display
                cached_result['from_cache'] = True
                return cached_result
            
            # Run normal analysis if not in cache
            st.info("No cached result found. Running fresh analysis...")
            result = self.find_statement_page_robust(pdf_content, statement_type, report_type)
            
            if result:
                # Save to cache
                self.save_to_metadata_cache(pdf_s3_path, symbol, period_end_date, report_type, statement_type, result, pdf_content)
                result['from_cache'] = False
                
            return result
            
        except Exception as e:
            st.error(f"Error in cached page finding: {e}")
            return None
        
    def setup_statement_aliases(self):
        """Setup statement type aliases with prioritization."""
        self.statement_aliases = {
            'balance_sheet': {
                'consolidated_group': [
                    'consolidated statement of financial position',
                    'group statement of financial position',
                    'consolidated balance sheet',
                    'group balance sheet'
                ],
                'standard': [
                    'statement of financial position',
                    'balance sheet'
                ]
            },
            'cashflow_statement': {
                'consolidated_group': [
                    'consolidated statement of cashflow',
                    'group statement of cashflow',
                    'consolidated cashflow statement',
                    'group cashflow statement',
                    'consolidated statement of cash flows',
                    'group statement of cash flows',
                    'consolidated cash flow statement',
                    'group cash flow statement'
                ],
                'standard': [
                    'statement of cashflow',
                    'cashflow statement',
                    'statement of cash flows',
                    'cash flow statement'
                ]
            },
            'income_statement': {
                'consolidated_group': [
                    'consolidated statement of profit and loss',
                    'group statement of profit and loss',
                    'consolidated income statement',
                    'group income statement',
                    'consolidated statement of comprehensive income',
                    'group statement of comprehensive income'
                ],
                'standard': [
                    'statement of profit and loss',
                    'income statement',
                    'statement of comprehensive income',
                    'profit and loss statement'
                ]
            }
        }
    
    def get_statement_search_terms(self, statement_type: str) -> Tuple[List[str], List[str]]:
        """Get prioritized search terms for a statement type."""
        statement_key = statement_type.lower().replace(' ', '_')
        
        if statement_key in self.statement_aliases:
            consolidated_terms = self.statement_aliases[statement_key]['consolidated_group']
            standard_terms = self.statement_aliases[statement_key]['standard']
            return consolidated_terms, standard_terms
        else:
            # If no aliases defined, use the original statement type
            return [statement_type], []
    
    def create_search_prompt_with_aliases(self, statement_type: str, report_type: str) -> str:
        """Create a search prompt that includes statement type aliases with prioritization."""
        consolidated_terms, standard_terms = self.get_statement_search_terms(statement_type)
        
        all_terms = consolidated_terms + standard_terms
        
        prompt = f"""
I need you to find a specific financial statement in this PDF. Follow these steps exactly:

TARGET: {statement_type}
REPORT TYPE: {report_type}

PRIORITY SEARCH TERMS (in order of preference):
1. HIGHEST PRIORITY - Consolidated/Group statements:
"""
        
        for term in consolidated_terms:
            prompt += f"   - {term}\n"
        
        if standard_terms:
            prompt += "\n2. FALLBACK - Standard statements (use only if consolidated/group not found):\n"
            for term in standard_terms:
                prompt += f"   - {term}\n"
        
        prompt += f"""
STEP 1: Create a page inventory
Go through the PDF sequentially and list what's on each page:
- Page 1 of PDF: [content description]
- Page 2 of PDF: [content description]  
- Page 3 of PDF: [content description]
...and so on

STEP 2: Identify the target statement with prioritization
Look for pages containing any of the target statement variations listed above. Follow this priority:
1. FIRST, search for consolidated/group versions (highest priority)
2. ONLY if no consolidated/group statements found, look for standard versions

Ignore:
- Cover pages
- Table of contents
- Auditor reports
- Notes to financial statements
- Directors' reports

Focus ONLY on the main financial statement that matches the target type.

STEP 3: Return the result
Count which sequential page position in the PDF contains your target statement.

CRITICAL: I need the PDF document page number (1st page = 1, 2nd page = 2, etc.), NOT any printed page number shown on the page.

Example: If the statement is on the 10th page when counting from the start of the PDF file, return pdf_page_position: 10 (even if that page shows "Page 8" in the corner).

Return ONLY this JSON format:

{{
  "pdf_page_position": [Sequential position in PDF starting from 1],
  "printed_page_number": "[Any printed page number visible on that page]",
  "statement_title": "[Exact statement title as it appears]",
  "statement_priority": "[consolidated/standard - which priority level was found]",
  "confidence": "[high/medium/low]",
  "reasoning": "[Explain: Found target statement on PDF page X, which shows printed page Y, using priority level]"
}}
"""
        return prompt

    def setup_aws_credentials(self):
        """Setup AWS credentials from environment variables."""
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        
    def setup_google_credentials(self):
        """Setup Google Cloud credentials."""
        self.project_id = os.getenv('PROJECT_ID', 'price-aggregator-f9e4b')
        self.vertex_region = os.getenv('VERTEX_REGION', 'us-east5')
        self.service_account_path = os.getenv('SERVICE_ACCOUNT_PATH')
        
        # Setup Google credentials
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive',
            'https://www.googleapis.com/auth/cloud-platform'
        ]
        
        try:
            if self.service_account_path and os.path.exists(self.service_account_path):
                self.credentials = service_account.Credentials.from_service_account_file(
                    self.service_account_path, scopes=scopes
                )
                st.success("Loaded credentials from service account file")
            else:
                service_account_json = os.getenv('SERVICE_ACCOUNT_JSON')
                if service_account_json:
                    service_account_info = json.loads(service_account_json)
                    self.credentials = service_account.Credentials.from_service_account_info(
                        service_account_info, scopes=scopes
                    )
                    st.success("Loaded credentials from environment variable")
                else:
                    st.error("No Google Cloud credentials found. Please set either:")
                    st.error("   - SERVICE_ACCOUNT_PATH environment variable pointing to your service account JSON file")
                    st.error("   - SERVICE_ACCOUNT_JSON environment variable with the JSON content")
                    raise ValueError("Google Cloud credentials not configured")
        except json.JSONDecodeError:
            st.error("Invalid JSON in SERVICE_ACCOUNT_JSON environment variable")
            raise
        except Exception as e:
            st.error(f"Error setting up Google credentials: {e}")
            raise
        
    def setup_clients(self):
        """Initialize AWS and Google clients."""
        try:
            if not self.aws_access_key_id or not self.aws_secret_access_key:
                st.error("AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
                raise ValueError("AWS credentials not configured")
            
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            
            self.textract_client = boto3.client(
                'textract',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            
            try:
                self.s3_client.list_buckets()
                st.success("AWS S3 connection successful")
            except Exception as e:
                st.error(f"AWS S3 connection failed: {e}")
                raise
            
            self.sheets_service = build('sheets', 'v4', credentials=self.credentials)
            st.success("Google Sheets connection successful")
            
            # Vertex AI endpoint
            self.vertex_endpoint = f"https://{self.vertex_region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.vertex_region}/publishers/anthropic/models/claude-3-5-sonnet-v2@20241022:streamRawPredict"
            
        except Exception as e:
            st.error(f"Error setting up clients: {e}")
            raise

    def parse_s3_path(self, s3_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse S3 path to extract bucket and key with enhanced debugging."""
        try:
            st.info(f"Parsing S3 path: '{s3_path}'")
            
            if not s3_path or not s3_path.strip():
                st.error("S3 path is empty or None")
                return None, None
            
            # Remove s3:// prefix if present
            original_path = s3_path
            if s3_path.startswith('s3://'):
                s3_path = s3_path[5:]  # Remove 's3://'
                st.info(f"Removed s3:// prefix: '{s3_path}'")
            
            # Split into bucket and key
            parts = s3_path.split('/', 1)
            st.info(f"Split parts: {parts}")
            
            if len(parts) == 2:
                bucket = parts[0]
                key = parts[1]
                st.info(f"Successfully parsed - Bucket: '{bucket}', Key: '{key}'")
                return bucket, key
            else:
                st.error(f"Invalid S3 path format. Expected format: 's3://bucket/key' or 'bucket/key'")
                st.error(f"Got: '{original_path}'")
                return None, None
                
        except Exception as e:
            st.error(f"Failed to parse S3 path '{s3_path}': {e}")
            return None, None
        
    def test_aws_connection(self):
        """Test AWS S3 connection and permissions."""
        try:
            st.subheader("AWS Connection Test")
            
            # Test basic S3 connection
            st.info("Testing S3 connection...")
            buckets = self.s3_client.list_buckets()
            st.success(f"S3 connection successful! Found {len(buckets['Buckets'])} buckets")
            
            # Show available buckets
            bucket_names = [bucket['Name'] for bucket in buckets['Buckets']]
            st.info(f"Available buckets: {bucket_names}")
            
            # Test specific bucket access
            test_bucket = 'jse-renamed-docs-copy'
            if test_bucket in bucket_names:
                st.info(f"Testing access to target bucket: {test_bucket}")
                try:
                    # Test bucket access
                    self.s3_client.head_bucket(Bucket=test_bucket)
                    st.success(f"Access to {test_bucket} confirmed!")
                    
                    # Test write permissions with a small test file
                    test_key = f"test/connection_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    test_content = "Connection test file"
                    
                    self.s3_client.put_object(
                        Bucket=test_bucket,
                        Key=test_key,
                        Body=test_content.encode('utf-8'),
                        ContentType='text/plain'
                    )
                    st.success("Write permission test successful!")
                    
                    # Clean up test file
                    self.s3_client.delete_object(Bucket=test_bucket, Key=test_key)
                    st.success("Test file cleaned up")
                    
                except Exception as bucket_error:
                    st.error(f"Error accessing bucket {test_bucket}: {bucket_error}")
            else:
                st.warning(f"Target bucket '{test_bucket}' not found in available buckets")
                
        except Exception as e:
            st.error(f"AWS connection test failed: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")

    def download_from_s3(self, s3_path: str) -> Optional[bytes]:
        """Download file from S3."""
        try:
            bucket, key = self.parse_s3_path(s3_path)
            if not bucket or not key:
                st.error(f"Invalid S3 path: {s3_path}")
                return None
            
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
        except ClientError as e:
            st.error(f"Failed to download from S3 - Bucket: {bucket}, Key: {key}, Error: {e}")
            return None

    def upload_to_s3(self, content: bytes, s3_path: str) -> bool:
        """Upload file to S3 with enhanced debugging."""
        try:
            st.info(f"Starting S3 upload process...")
            st.info(f"Content size: {len(content)} bytes")
            st.info(f"S3 path: {s3_path}")
            
            bucket, key = self.parse_s3_path(s3_path)
            if not bucket or not key:
                st.error(f"Invalid S3 path for upload: {s3_path}")
                st.error(f"Parsed bucket: {bucket}, key: {key}")
                return False
            
            st.info(f"Parsed S3 details - Bucket: {bucket}, Key: {key}")
            
            # Test S3 connection first
            try:
                st.info("Testing S3 connection...")
                self.s3_client.head_bucket(Bucket=bucket)
                st.success("S3 bucket connection successful")
            except Exception as conn_error:
                st.error(f"S3 bucket connection failed: {conn_error}")
                return False
            
            # Attempt the upload with detailed progress
            st.info("Starting file upload...")
            
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=content,
                ContentType='text/csv',
                Metadata={
                    'uploaded_by': 'financial_validation_app',
                    'upload_timestamp': datetime.now().isoformat()
                }
            )
            
            st.success(f"Successfully uploaded to S3!")
            st.success(f"Bucket: {bucket}")
            st.success(f"Key: {key}")
            st.success(f"Full path: {s3_path}")
            
            # Verify the upload
            try:
                st.info("Verifying upload...")
                response = self.s3_client.head_object(Bucket=bucket, Key=key)
                file_size = response.get('ContentLength', 0)
                st.success(f"Upload verified! File size: {file_size} bytes")
                return True
            except Exception as verify_error:
                st.warning(f"Upload succeeded but verification failed: {verify_error}")
                return True  # Still return True since upload succeeded
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            st.error(f"AWS ClientError: {error_code} - {error_message}")
            st.error(f"Full error details: {e}")
            return False
        except Exception as e:
            st.error(f"Unexpected error during S3 upload: {e}")
            st.error(f"Error type: {type(e).__name__}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return False

    def upload_to_s3_with_custom_path(self, csv_content: str, custom_s3_path: str) -> bool:
        """
        Upload CSV content to S3 using a custom path provided by the user.
        
        Args:
            csv_content (str): The CSV content to upload
            custom_s3_path (str): Full S3 path including bucket (e.g., s3://bucket/path/file.csv)
        
        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            # Validate CSV content
            if not csv_content or not csv_content.strip():
                st.error("No CSV content provided for upload")
                return False
            
            # Validate S3 path format
            if not custom_s3_path.startswith('s3://'):
                st.error("S3 path must start with 's3://'")
                return False
            
            # Test if content is valid CSV
            try:
                test_df = pd.read_csv(io.StringIO(csv_content))
                if len(test_df) == 0:
                    st.warning("CSV content appears to be empty (no data rows)")
                else:
                    st.info(f"CSV validation passed: {len(test_df)} rows, {len(test_df.columns)} columns")
            except Exception as csv_error:
                st.error(f"Invalid CSV format: {csv_error}")
                return False
            
            # Parse the custom S3 path
            bucket, key = self.parse_s3_path(custom_s3_path)
            if not bucket or not key:
                st.error(f"Invalid S3 path format: {custom_s3_path}")
                return False
            
            st.info(f"Uploading to S3 bucket: {bucket}, key: {key}")
            
            # Convert CSV string to bytes
            csv_bytes = csv_content.encode('utf-8')
            
            # Upload to S3
            st.info(f"Uploading CSV ({len(csv_bytes)} bytes) to S3...")
            
            success = self.upload_to_s3(csv_bytes, custom_s3_path)
            
            if success:
                st.success(f"CSV uploaded successfully!")
                st.success(f"Location: {custom_s3_path}")
                return True
            else:
                st.error("S3 upload failed")
                return False
                
        except Exception as e:
            st.error(f"Error in custom S3 upload: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return False

    def upload_csv_to_s3(self, csv_content: str, symbol: str, statement_type: str, 
                         period_end_date: str, report_type: str, 
                         statement_title: str = None) -> tuple[bool, str]:
        """
        Upload CSV content to S3 with proper validation and path generation.
        
        Returns:
            tuple: (success: bool, s3_path: str)
        """
        try:
            # Validate CSV content
            if not csv_content or not csv_content.strip():
                st.error("No CSV content provided for upload")
                return False, ""
            
            # Test if content is valid CSV
            try:
                test_df = pd.read_csv(io.StringIO(csv_content))
                if len(test_df) == 0:
                    st.warning("CSV content appears to be empty (no data rows)")
                else:
                    st.info(f"CSV validation passed: {len(test_df)} rows, {len(test_df.columns)} columns")
            except Exception as csv_error:
                st.error(f"Invalid CSV format: {csv_error}")
                st.text_area("Raw content that failed validation:", csv_content[:500], height=150)
                return False, ""
            
            # Generate statement title for filename if not provided
            if not statement_title:
                statement_title = self.get_statement_filename_part(statement_type)
            else:
                # Clean the provided statement title for filename use
                statement_title = statement_title.lower().replace(' ', '_').replace('/', '_').replace('&', 'and')
            
            # Generate S3 path
            s3_path = self.generate_s3_path(
                symbol=symbol,
                statement_title=statement_title,
                period_end_date=period_end_date,
                report_type=report_type,
                statement_type=report_type  # Pass audit status as statement_type parameter
            )
            
            st.info(f"Generated S3 path: {s3_path}")
            
            # Convert CSV string to bytes
            csv_bytes = csv_content.encode('utf-8')
            
            # Upload to S3
            st.info(f"Uploading CSV ({len(csv_bytes)} bytes) to S3...")
            
            success = self.upload_to_s3(csv_bytes, s3_path)
            
            if success:
                st.success(f"CSV uploaded successfully!")
                st.success(f"Location: {s3_path}")
                return True, s3_path
            else:
                st.error("S3 upload failed")
                return False, s3_path
                
        except Exception as e:
            st.error(f"Error in CSV upload process: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return False, ""

    def extract_statement_data_only(self, pdf_content: bytes, selected_pages: list, 
                                   statement_type: str, report_type: str, 
                                   extraction_method: str) -> tuple[bool, str]:
        """
        Extract statement data only (separated from upload).
        
        Returns:
            tuple: (success: bool, csv_content: str)
        """
        try:
            pages_text = ", ".join(map(str, selected_pages))
            
            # Display extraction info
            st.info(f"**Extracting:** {statement_type}")
            st.info(f"**Report Type:** {report_type}")
            st.info(f"**Pages:** {pages_text}")
            st.info(f"**Method:** {extraction_method}")
            
            # Show expected statement titles for validation
            consolidated_terms, standard_terms = self.get_statement_search_terms(statement_type)
            all_terms = consolidated_terms + standard_terms
            
            with st.expander("Expected Statement Titles", expanded=False):
                st.write("The extraction will look for statements with these titles:")
                for term in all_terms:
                    st.write(f"• {term}")
            
            # Perform extraction
            with st.spinner(f"Extracting {statement_type} from {report_type} on page(s) {pages_text}..."):
                if extraction_method == "Claude Sonnet 4 (Vertex AI)":
                    extracted_csv = self.extract_with_claude(
                        pdf_content=pdf_content,
                        page_numbers=selected_pages,
                        statement_type=statement_type,
                        report_type=report_type
                    )
                else:  # AWS Textract
                    extracted_csv = self.extract_with_textract(
                        pdf_content=pdf_content,
                        page_numbers=selected_pages,
                        statement_type=statement_type,
                        report_type=report_type
                    )
            
            if not extracted_csv or not extracted_csv.strip():
                st.error("Extraction failed - no data returned")
                return False, ""
            
            st.success("Extraction completed!")
            
            # Additional validation for Textract
            if extraction_method == "AWS Textract":
                st.warning("**Manual Verification Required**: Please verify that the extracted data corresponds to the selected statement type.")
                st.info(f"Expected: **{statement_type}** from **{report_type}**")
            
            return True, extracted_csv
            
        except Exception as e:
            st.error(f"Error in extraction process: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return False, ""

    def analyze_all_statements_in_pdf(self, pdf_content: bytes, symbol: str, period_end_date: str) -> Dict:
        """Analyze PDF to find all available statement types and their pages."""
        try:
            # Get total page count
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            total_pages = len(doc)
            doc.close()
            
            st.info(f"Analyzing PDF for all statement types ({total_pages} pages)...")
            
            # Define all possible statement types to search for
            statement_types_to_search = ['balance_sheet', 'cashflow_statement', 'income_statement']
            
            all_statements_found = {}
            
            for stmt_type in statement_types_to_search:
                st.info(f"Searching for {stmt_type.replace('_', ' ').title()}...")
                
                # Use the robust method to find this statement type
                result = self.find_statement_page_robust(pdf_content, stmt_type, "Annual Report")
                
                if result:
                    # Handle both single and multi-page results
                    page_positions = result.get('pdf_page_positions', [])
                    if not page_positions:
                        # Fallback to single page format
                        single_page = result.get('pdf_page_position', None)
                        if single_page:
                            page_positions = [single_page]
                    
                    if page_positions:
                        all_statements_found[stmt_type] = {
                            'pages': page_positions,
                            'printed_pages': result.get('printed_page_numbers', []),
                            'title': result.get('statement_title', stmt_type.replace('_', ' ').title()),
                            'priority': result.get('statement_priority', 'standard'),
                            'confidence': result.get('confidence', 'medium'),
                            'reasoning': result.get('reasoning', ''),
                            'is_multi_page': len(page_positions) > 1
                        }
                        
                        priority_icon = "⭐" if result.get('statement_priority') == 'consolidated' else "📄"
                        st.success(f"Found {stmt_type.replace('_', ' ').title()} {priority_icon} on pages: {', '.join(map(str, page_positions))}")
                    else:
                        st.warning(f"No pages found for {stmt_type.replace('_', ' ').title()}")
                else:
                    st.warning(f"Could not locate {stmt_type.replace('_', ' ').title()}")
            
            return all_statements_found
            
        except Exception as e:
            st.error(f"Error analyzing PDF for all statements: {e}")
            return {}

    def display_content_page(self, pdf_content: bytes, all_statements: Dict, symbol: str, period_end_date: str, report_type: str):
        """Display a content page with links to all found statements."""
        try:
            st.header("📋 PDF Content Overview")
            st.subheader(f"Available Financial Statements for {symbol}")
            st.info(f"Period: {period_end_date} | Report Type: {report_type}")
            
            if not all_statements:
                st.warning("No financial statements were found in this PDF.")
                return
            
            # Sort statements by priority (consolidated first)
            sorted_statements = sorted(
                all_statements.items(), 
                key=lambda x: (0 if x[1]['priority'] == 'consolidated' else 1, x[0])
            )
            
            st.write("Click on any statement type below to view and extract that statement:")
            st.write("---")
            
            # Create columns for better layout
            col1, col2 = st.columns([1, 1])
            
            for i, (stmt_type, info) in enumerate(sorted_statements):
                # Alternate between columns
                with col1 if i % 2 == 0 else col2:
                    # Create an expandable section for each statement
                    with st.expander(f"{self.get_statement_display_name(stmt_type)} {self.get_priority_icon(info['priority'])}", expanded=False):
                        
                        # Statement details
                        st.write(f"**Statement Title:** {info['title']}")
                        st.write(f"**Pages:** {', '.join(map(str, info['pages']))}")
                        if info['printed_pages']:
                            st.write(f"**Printed Pages:** {', '.join(map(str, info['printed_pages']))}")
                        st.write(f"**Priority:** {info['priority'].title()}")
                        st.write(f"**Confidence:** {info['confidence'].title()}")
                        if info['is_multi_page']:
                            st.write("**Type:** Multi-page statement")
                        
                        # Show page previews
                        if len(info['pages']) == 1:
                            # Single page preview
                            img_data = self.render_pdf_page(pdf_content, info['pages'][0] - 1)
                            if img_data:
                                st.image(img_data, caption=f"Page {info['pages'][0]}", use_column_width=True)
                        else:
                            # Multi-page preview (show first 2 pages)
                            for page_num in info['pages'][:2]:
                                img_data = self.render_pdf_page(pdf_content, page_num - 1)
                                if img_data:
                                    st.image(img_data, caption=f"Page {page_num}", use_column_width=True)
                            if len(info['pages']) > 2:
                                st.write(f"... and {len(info['pages']) - 2} more page(s)")
                        
                        # Extract button for this specific statement
                        if st.button(f"Extract {self.get_statement_display_name(stmt_type)}", 
                                   key=f"extract_{stmt_type}_{symbol}_{period_end_date}",
                                   type="primary"):
                            # Store the selected statement info in session state
                            st.session_state.selected_statement_for_extraction = {
                                'statement_type': stmt_type,
                                'pages': info['pages'],
                                'title': info['title'],
                                'priority': info['priority'],
                                'symbol': symbol,
                                'period_end_date': period_end_date,
                                'report_type': report_type
                            }
                            st.rerun()
            
            st.write("---")
            st.info("💡 **Tip:** Statements marked with ⭐ are consolidated/group statements (preferred). Statements marked with 📄 are standard statements.")
            
        except Exception as e:
            st.error(f"Error displaying content page: {e}")

    def get_statement_filename_part(self, stmt_type: str) -> str:
        """Get filename-friendly part for statement types."""
        filename_parts = {
            'balance_sheet': 'statement_of_financial_position',
            'cashflow_statement': 'cashflow_statement', 
            'income_statement': 'income_statement'
        }
        return filename_parts.get(stmt_type, stmt_type.replace(' ', '_').lower())

    def get_statement_display_name(self, stmt_type: str) -> str:
        """Get a user-friendly display name for statement types.""" 
        display_names = {
            'balance_sheet': 'Balance Sheet / Statement of Financial Position',
            'cashflow_statement': 'Cash Flow Statement',
            'income_statement': 'Income Statement / Profit & Loss'
        }
        return display_names.get(stmt_type, stmt_type.replace('_', ' ').title())

    def get_priority_icon(self, priority: str) -> str:
        """Get priority icon for statement types."""
        if priority == 'consolidated':
            return "⭐"
        else:
            return "📄"

    def handle_statement_extraction(self, pdf_content: bytes):
        """Handle extraction of a selected statement from the content page."""
        try:
            if 'selected_statement_for_extraction' not in st.session_state:
                return False
            
            extraction_info = st.session_state.selected_statement_for_extraction
            
            st.header("🔄 Statement Extraction")
            st.subheader(f"Extracting: {self.get_statement_display_name(extraction_info['statement_type'])}")
            
            # Display extraction details
            st.info(f"**Statement Type:** {self.get_statement_display_name(extraction_info['statement_type'])}")
            st.info(f"**Pages:** {', '.join(map(str, extraction_info['pages']))}")
            st.info(f"**Priority:** {extraction_info['priority'].title()} {self.get_priority_icon(extraction_info['priority'])}")
            st.info(f"**Statement Title:** {extraction_info['title']}")
            
            # Show page previews
            st.subheader("Page Preview(s)")
            for page_num in extraction_info['pages'][:3]:  # Show max 3 pages
                img_data = self.render_pdf_page(pdf_content, page_num - 1)
                if img_data:
                    st.image(img_data, caption=f"Page {page_num}", use_column_width=True)
            
            if len(extraction_info['pages']) > 3:
                st.info(f"... and {len(extraction_info['pages']) - 3} more page(s)")
            
            # Extraction method selection
            st.subheader("Extraction Method")
            extraction_method = st.radio(
                "Choose extraction method:",
                ["Claude Sonnet 4 (Vertex AI)", "AWS Textract"],
                key="content_page_extraction_method"
            )
            
            # Get statement aliases for validation
            consolidated_terms, standard_terms = self.get_statement_search_terms(extraction_info['statement_type'])
            all_terms = consolidated_terms + standard_terms
            
            with st.expander("Expected Statement Titles", expanded=False):
                st.write("The extraction will look for statements with these titles:")
                for term in all_terms:
                    st.write(f"• {term}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Extract Financial Data", type="primary", key="extract_from_content_page"):
                    pages_text = ", ".join(map(str, extraction_info['pages']))
                    
                    # Use the separated extraction method
                    success, extracted_csv = self.extract_statement_data_only(
                        pdf_content=pdf_content,
                        selected_pages=extraction_info['pages'],
                        statement_type=extraction_info['statement_type'],
                        report_type=extraction_info['report_type'],
                        extraction_method=extraction_method
                    )
                    
                    if success and extracted_csv:
                        st.success("Extraction completed!")
                        
                        # Store the extracted data in session state
                        st.session_state.extracted_csv_data = extracted_csv
                        st.session_state.extraction_details = extraction_info
                        st.rerun()
                    else:
                        st.error("Extraction failed")
            
            with col2:
                if st.button("← Back to Content Overview", key="back_to_content"):
                    if 'selected_statement_for_extraction' in st.session_state:
                        del st.session_state.selected_statement_for_extraction
                    st.rerun()
            
            return True
            
        except Exception as e:
            st.error(f"Error handling statement extraction: {e}")
            return False

    def display_extracted_data_from_content_page(self):
        """Display extracted data and handle file operations."""
        try:
            if 'extracted_csv_data' not in st.session_state or 'extraction_details' not in st.session_state:
                return False
            
            extracted_csv = st.session_state.extracted_csv_data
            extraction_info = st.session_state.extraction_details
            
            st.header("📊 Extraction Results")
            
            # Display extracted data
            st.subheader("Extracted CSV Data")
            
            try:
                extracted_df = pd.read_csv(io.StringIO(extracted_csv))
                st.dataframe(extracted_df, use_container_width=True)
                
                # Show data validation info
                if len(extracted_df) > 0:
                    st.info(f"Extracted {len(extracted_df)} rows of data")
                else:
                    st.warning("No data rows extracted - please verify the page selection")
            except Exception as e:
                st.text_area("Raw CSV Data:", extracted_csv, height=400)
                st.warning("Could not parse as CSV - showing raw data above")
            
            # File naming and upload section
            st.subheader("File Operations")
            
            statement_title = extraction_info['title']
            symbol = extraction_info['symbol']
            period_end_date = extraction_info['period_end_date']
            report_type = extraction_info['report_type']
            
            # Generate auto filename
            auto_filename = f"{symbol.lower()}-{statement_title.lower().replace(' ', '_')}-{self.format_period_end_date(period_end_date)}.csv"
            
            filename = st.text_input(
                "CSV Filename:",
                value=auto_filename,
                key="content_page_filename"
            )
            
            # Generate S3 path
            s3_path = self.generate_s3_path(
                symbol,
                statement_title,
                period_end_date,
                report_type,
                report_type  # Pass the audit status
            )
            
            st.info(f"S3 Upload Path: {s3_path}")
            
            # Separate buttons for different operations
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download button
                st.download_button(
                    label="Download CSV File",
                    data=extracted_csv,
                    file_name=filename,
                    mime="text/csv",
                    key="download_extracted_csv"
                )
            
            with col2:
                # Upload to S3 button
                if st.button("Upload to S3", type="primary", key="upload_from_content_page"):
                    with st.spinner("Uploading to S3..."):
                        success, upload_s3_path = self.upload_csv_to_s3(
                            csv_content=extracted_csv,
                            symbol=symbol,
                            statement_type=extraction_info['statement_type'],
                            period_end_date=period_end_date,
                            report_type=report_type,
                            statement_title=statement_title
                        )
                        
                        if success:
                            st.success(f"File uploaded successfully to: {upload_s3_path}")
                            # Store the successful upload path
                            st.session_state.uploaded_s3_path = upload_s3_path
                        else:
                            st.error("Failed to upload file to S3")
            
            with col3:
                # Back button
                if st.button("← Back to Content Overview", key="back_to_content_from_results"):
                    # Clear the extraction results but keep the content page
                    if 'extracted_csv_data' in st.session_state:
                        del st.session_state.extracted_csv_data
                    if 'extraction_details' in st.session_state:
                        del st.session_state.extraction_details
                    if 'selected_statement_for_extraction' in st.session_state:
                        del st.session_state.selected_statement_for_extraction
                    st.rerun()
            
            # Show upload success message and offer spreadsheet update
            if 'uploaded_s3_path' in st.session_state:
                st.success("✅ File successfully uploaded to S3!")
                st.info(f"📍 Location: {st.session_state.uploaded_s3_path}")
                
                # Offer to update spreadsheet
                if st.button("Update Google Spreadsheet", key="update_spreadsheet"):
                    with st.spinner("Updating spreadsheet..."):
                        # Update spreadsheet logic would go here
                        # For now, just show success
                        st.success("📊 Spreadsheet update feature coming soon!")
                        # Clear the uploaded path indicator
                        del st.session_state.uploaded_s3_path
            
            return True
            
        except Exception as e:
            st.error(f"Error displaying extracted data: {e}")
            return False

    def get_access_token(self) -> str:
        """Get access token for Vertex AI API."""
        if not self.credentials.valid:
            self.credentials.refresh(Request())
        return self.credentials.token

    def find_statement_page(self, pdf_content: bytes, statement_type: str, report_type: str) -> Dict:
        """Use Claude to find the correct page containing the target statement with aliases."""
        try:
            pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
            
            # Create prompt with aliases and prioritization
            prompt = self.create_search_prompt_with_aliases(statement_type, report_type)
            
            access_token = self.get_access_token()
            
            payload = {
                "anthropic_version": "vertex-2023-10-16",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0
            }
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                self.vertex_endpoint,
                headers=headers,
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                response_data = response.json()
                if 'content' in response_data and len(response_data['content']) > 0:
                    response_text = response_data['content'][0]['text']
                    
                    # Extract JSON from response
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        return json.loads(json_str)
            
            return None
                
        except Exception as e:
            st.error(f"Error creating detailed page mapping: {e}")
            return None

    def extract_pdf_chunk(self, pdf_content: bytes, start_page: int, end_page: int) -> Optional[bytes]:
        """Extract a chunk of pages from the PDF."""
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            
            # Create new PDF with just the chunk
            chunk_doc = fitz.open()
            chunk_doc.insert_pdf(doc, from_page=start_page-1, to_page=end_page-1)
            chunk_pdf = chunk_doc.tobytes()
            
            doc.close()
            chunk_doc.close()
            
            return chunk_pdf
            
        except Exception as e:
            st.error(f"Error extracting PDF chunk: {e}")
            return None

    def extract_with_claude(self, pdf_content: bytes, page_numbers: List[int], statement_type: str, report_type: str = None) -> Optional[str]:
        """Extract financial data from multiple pages using Claude and consolidate into one CSV."""
        try:
            pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
            
            if isinstance(page_numbers, int):
                page_numbers = [page_numbers]
            
            pages_text = ", ".join([str(p) for p in page_numbers])
            
            # Ensure page_numbers is a list of integers
            page_numbers = [int(p) for p in page_numbers]
            
            # Get all possible search terms for better extraction
            consolidated_terms, standard_terms = self.get_statement_search_terms(statement_type)
            all_terms = consolidated_terms + standard_terms
            
            search_terms_text = ", ".join(all_terms)
            
            prompt = f"""
Extract financial data from pages {pages_text} of this PDF document and consolidate into a single CSV.

Target Statement Type: {statement_type}
Pages to extract: {page_numbers}
Possible Statement Titles: {search_terms_text}

Instructions:
1. Extract data from ALL specified pages: {page_numbers}
2. If the statement spans multiple pages, consolidate all data into a single coherent table
3. Remove any duplicate headers that appear on continuation pages
4. Maintain the hierarchical structure of the statement across all pages
5. Include line item names and all numerical values from all pages
6. Return consolidated data as CSV format with these columns:
   - line_item: The name/description of the financial line item
   - current_period: Current period value (remove currency symbols, keep numbers only)
   - prior_period: Prior period value if available (remove currency symbols, keep numbers only)
   - notes: Any note references (like Note 1, Note 2, etc.)

For multi-page statements:
- Combine all line items in their proper order
- Ensure no data is duplicated
- Maintain the logical flow of the statement structure

Return only the consolidated CSV data with headers, no other text.
"""
            
            access_token = self.get_access_token()
            
            payload = {
                "anthropic_version": "vertex-2023-10-16",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": 4000,
                "temperature": 0
            }
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                self.vertex_endpoint,
                headers=headers,
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                response_data = response.json()
                if 'content' in response_data and len(response_data['content']) > 0:
                    return response_data['content'][0]['text']
            else:
                st.error(f"Claude API request failed: {response.status_code}")
                return None
                
        except Exception as e:
            st.error(f"Error with Claude extraction: {e}")
            return None

    def extract_with_textract(self, pdf_content: bytes, page_numbers: List[int], statement_type: str = None, report_type: str = None) -> Optional[str]:
        """Extract financial data from multiple pages using AWS Textract and consolidate into one CSV."""
        try:
            if isinstance(page_numbers, int):
                page_numbers = [page_numbers]
            
            # Display what we're extracting
            pages_text = ", ".join([str(p) for p in page_numbers])
            st.info(f"Statement Type: **{statement_type or 'Unknown Statement Type'}**")
            st.info(f"Report Type: **{report_type or 'Not specified'}**")
            st.info(f"Pages: **{pages_text}**")
            st.info(f"Method: **AWS Textract**")
            st.warning("Note: Textract extracts table structure but cannot validate statement type. Please verify the extracted data matches your selected statement type.")
            
            all_tables = []
            
            for page_number in page_numbers:
                st.info(f"Processing page {page_number} with Textract...")
                
                # Extract specific page from PDF
                doc = fitz.open(stream=pdf_content, filetype="pdf")
                if page_number > len(doc):
                    st.error(f"Page {page_number} not found in PDF")
                    continue
                    
                # Create a new PDF with just the target page
                single_page_doc = fitz.open()
                single_page_doc.insert_pdf(doc, from_page=page_number-1, to_page=page_number-1)
                single_page_pdf = single_page_doc.tobytes()
                doc.close()
                single_page_doc.close()
                
                # Upload single page to S3 for Textract
                temp_bucket = os.getenv('TEMP_BUCKET', 'jse-renamed-docs-copy')
                temp_s3_key = f"temp/textract_page_{page_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                
                self.s3_client.put_object(
                    Bucket=temp_bucket, 
                    Key=temp_s3_key, 
                    Body=single_page_pdf,
                    ContentType='application/pdf'
                )
                
                # Start document analysis
                response = self.textract_client.start_document_analysis(
                    DocumentLocation={
                        'S3Object': {
                            'Bucket': temp_bucket,
                            'Name': temp_s3_key
                        }
                    },
                    FeatureTypes=['TABLES']
                )
                
                job_id = response['JobId']
                
                # Poll for completion
                max_wait_time = 300
                wait_time = 0
                
                while wait_time < max_wait_time:
                    result = self.textract_client.get_document_analysis(JobId=job_id)
                    status = result['JobStatus']
                    
                    if status == 'SUCCEEDED':
                        break
                    elif status == 'FAILED':
                        st.error(f"Textract analysis failed for page {page_number}")
                        break
                    elif status == 'IN_PROGRESS':
                        time.sleep(10)
                        wait_time += 10
                    else:
                        time.sleep(5)
                        wait_time += 5
                
                if wait_time >= max_wait_time:
                    st.error(f"Textract analysis timed out for page {page_number}")
                    continue
                
                if status == 'SUCCEEDED':
                    # Process results for this page
                    page_tables = self.extract_tables_from_blocks(result.get('Blocks', []))
                    if page_tables:
                        # Get the largest table from this page
                        best_table = max(page_tables, key=lambda t: len(t))
                        all_tables.append((page_number, best_table))
                        st.success(f"Extracted table data from page {page_number}")
                    else:
                        st.warning(f"No tables found on page {page_number}")
                
                # Clean up temp file
                try:
                    self.s3_client.delete_object(Bucket=temp_bucket, Key=temp_s3_key)
                except:
                    pass
            
            if not all_tables:
                st.error("No tables found on any of the specified pages")
                return None
            
            # Consolidate all tables into one CSV
            result_csv = self.consolidate_tables_to_csv(all_tables)
            
            if result_csv:
                st.success(f"Successfully extracted table data from **{report_type or 'report'}** on pages {pages_text}")
                if statement_type:
                    st.info(f"Please verify that the extracted data corresponds to: **{statement_type}**")
                return result_csv
            else:
                st.error("Failed to consolidate extracted tables")
                return None
                
        except Exception as e:
            st.error(f"Error with Textract extraction: {e}")
            return None

    def consolidate_tables_to_csv(self, page_tables: List[Tuple[int, List[List[str]]]]) -> str:
        """Consolidate multiple page tables into a single CSV."""
        try:
            consolidated_rows = []
            headers_added = False
            
            for page_number, table_data in page_tables:
                for row_idx, row in enumerate(table_data):
                    # Skip empty rows
                    if not any(cell.strip() for cell in row):
                        continue
                    
                    # For first page or if headers not added yet, include header row
                    if not headers_added and row_idx == 0:
                        # Check if this looks like a header row (contains common header words)
                        row_text = ' '.join(row).lower()
                        if any(word in row_text for word in ['assets', 'liabilities', 'revenue', 'expenses', 'line', 'item', 'amount', 'note']):
                            consolidated_rows.append(row)
                            headers_added = True
                        else:
                            # Add this row as data
                            consolidated_rows.append(row)
                    else:
                        # Skip header rows on subsequent pages
                        row_text = ' '.join(row).lower()
                        is_likely_header = (
                            row_idx == 0 and 
                            any(word in row_text for word in ['assets', 'liabilities', 'revenue', 'expenses', 'line', 'item', 'amount', 'note']) and
                            not any(char.isdigit() for char in row_text)
                        )
                        
                        if not is_likely_header:
                            consolidated_rows.append(row)
            
            # Convert to CSV
            if not consolidated_rows:
                return ""
            
            # Ensure all rows have the same number of columns
            max_cols = max(len(row) for row in consolidated_rows) if consolidated_rows else 0
            
            csv_lines = []
            for row in consolidated_rows:
                # Pad row to max columns
                padded_row = row + [''] * (max_cols - len(row))
                
                clean_row = []
                for cell in padded_row:
                    clean_cell = str(cell).strip().replace('\n', ' ').replace('\r', '')
                    if ',' in clean_cell or '"' in clean_cell or '\n' in clean_cell:
                        clean_cell = '"' + clean_cell.replace('"', '""') + '"'
                    clean_row.append(clean_cell)
                
                csv_lines.append(','.join(clean_row))
            
            return '\n'.join(csv_lines)
            
        except Exception as e:
            st.error(f"Error consolidating tables: {e}")
            return ""

    def extract_tables_from_blocks(self, blocks):
        """Extract table data from Textract blocks."""
        try:
            block_map = {block['Id']: block for block in blocks}
            table_blocks = [block for block in blocks if block['BlockType'] == 'TABLE']
            
            tables = []
            
            for table_block in table_blocks:
                rows_count = 0
                cols_count = 0
                
                if 'Relationships' in table_block:
                    for relationship in table_block['Relationships']:
                        if relationship['Type'] == 'CHILD':
                            for child_id in relationship['Ids']:
                                child_block = block_map.get(child_id)
                                if child_block and child_block['BlockType'] == 'CELL':
                                    rows_count = max(rows_count, child_block.get('RowIndex', 0))
                                    cols_count = max(cols_count, child_block.get('ColumnIndex', 0))
                
                table_data = [['' for _ in range(cols_count)] for _ in range(rows_count)]
                
                if 'Relationships' in table_block:
                    for relationship in table_block['Relationships']:
                        if relationship['Type'] == 'CHILD':
                            for child_id in relationship['Ids']:
                                child_block = block_map.get(child_id)
                                if child_block and child_block['BlockType'] == 'CELL':
                                    row_idx = child_block.get('RowIndex', 1) - 1
                                    col_idx = child_block.get('ColumnIndex', 1) - 1
                                    
                                    cell_text = self.get_cell_text(child_block, block_map)
                                    
                                    if 0 <= row_idx < rows_count and 0 <= col_idx < cols_count:
                                        table_data[row_idx][col_idx] = cell_text
                
                tables.append(table_data)
            
            return tables
            
        except Exception as e:
            st.error(f"Error extracting tables: {e}")
            return []

    def get_cell_text(self, cell_block, block_map):
        """Extract text from a cell block."""
        try:
            text_parts = []
            
            if 'Relationships' in cell_block:
                for relationship in cell_block['Relationships']:
                    if relationship['Type'] == 'CHILD':
                        for child_id in relationship['Ids']:
                            child_block = block_map.get(child_id)
                            if child_block and child_block['BlockType'] == 'WORD':
                                text_parts.append(child_block.get('Text', ''))
            
            return ' '.join(text_parts).strip()
            
        except Exception as e:
            return ''

    def table_to_csv(self, table_data):
        """Convert table data to CSV format."""
        try:
            csv_lines = []
            for row in table_data:
                clean_row = []
                for cell in row:
                    clean_cell = str(cell).strip().replace('\n', ' ').replace('\r', '')
                    if ',' in clean_cell or '"' in clean_cell or '\n' in clean_cell:
                        clean_cell = '"' + clean_cell.replace('"', '""') + '"'
                    clean_row.append(clean_cell)
                
                csv_lines.append(','.join(clean_row))
            
            return '\n'.join(csv_lines)
            
        except Exception as e:
            st.error(f"Error converting table to CSV: {e}")
            return ""

    def render_pdf_page(self, pdf_content: bytes, page_num: int = 0) -> bytes:
        """Render a PDF page as an image."""
        try:
            if not pdf_content:
                return None
                
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            
            if page_num >= len(doc):
                doc.close()
                return None
                
            page = doc[page_num]
            
            # Try different matrix scales if default fails
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
            except:
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))  # Normal resolution
                except Exception as pix_error:
                    doc.close()
                    return None
            
            img_data = pix.tobytes("png")
            
            doc.close()
            return img_data
            
        except Exception as e:
            return None

    def format_period_end_date(self, date_str: str) -> str:
        """Format period end date to month-dd-yyyy format."""
        try:
            # Try to parse the date string
            if pd.isna(date_str) or not date_str:
                return ""
            
            # Handle various date formats
            date_str = str(date_str).strip()
            
            # If already in desired format, return as-is
            if re.match(r'[a-zA-Z]+-\d{2}-\d{4}', date_str):
                return date_str.lower()
            
            # Try to parse common formats
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    month_name = dt.strftime('%B').lower()
                    return f"{month_name}-{dt.day:02d}-{dt.year}"
                except:
                    continue
            
            return date_str.lower().replace('/', '-').replace(' ', '-')
            
        except Exception as e:
            return str(date_str).lower()

    def generate_s3_path(self, symbol: str, statement_title: str, period_end_date: str, report_type: str, statement_type: str = None) -> str:
        """Generate S3 path based on the specified format."""
        try:
            # Clean symbol
            clean_symbol = symbol.upper()
            
            # Determine if audited or unaudited based on statement_type (not report_type)
            if statement_type and statement_type.lower() == 'unaudited':
                folder_type = "unaudited_financial_statements"
            elif statement_type and statement_type.lower() == 'audited':
                folder_type = "audited_financial_statements"
            else:
                # Fallback to original logic if statement_type not provided
                folder_type = "unaudited_financial_statements" if "quarterly" in report_type.lower() else "audited_financial_statements"
            
            # Extract year from period_end_date
            year = "2024"  # Default
            if period_end_date:
                try:
                    # Extract year from various formats
                    year_match = re.search(r'\d{4}', str(period_end_date))
                    if year_match:
                        year = year_match.group()
                except:
                    pass
            
            # Clean statement title
            clean_title = statement_title.lower()
            clean_title = re.sub(r'[^\w\s-]', '', clean_title)  # Remove special chars
            clean_title = re.sub(r'\s+', '_', clean_title.strip())  # Replace spaces with underscores
            
            # Format period end date
            formatted_date = self.format_period_end_date(period_end_date)
            
            # Generate filename
            filename = f"{clean_symbol.lower()}-{clean_title}-{formatted_date}.csv"
            
            # Generate full S3 path
            s3_path = f"s3://jse-renamed-docs-copy/CSV-Copy/{clean_symbol}/{folder_type}/{year}/{filename}"
            
            return s3_path
            
        except Exception as e:
            st.error(f"Error generating S3 path: {e}")
            return f"s3://jse-renamed-docs-copy/CSV-Copy/{symbol}/audited_financial_statements/2024/{symbol}-statement-{period_end_date}.csv"

    def add_row_to_spreadsheet(self, spreadsheet_id: str, tab_name: str, new_row_data: Dict):
        """Add a new row to the Google Sheet."""
        try:
            # Get current headers
            result = self.sheets_service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=f"'{tab_name}'!1:1"
            ).execute()
            
            headers = result.get('values', [[]])[0]
            
            # Prepare new row with all columns
            new_row = []
            for header in headers:
                value = new_row_data.get(header, '')
                new_row.append(value)
            
            # Append the row
            self.sheets_service.spreadsheets().values().append(
                spreadsheetId=spreadsheet_id,
                range=f"'{tab_name}'",
                valueInputOption='RAW',
                body={'values': [new_row]}
            ).execute()
            
            return True
            
        except Exception as e:
            st.error(f"Failed to add row to spreadsheet: {e}")
            return False


# Initialize the app
@st.cache_resource
def get_app():
    try:
        return FinancialValidationApp()
    except Exception as e:
        st.error(f"Failed to initialize app: {e}")
        st.error("Please check your .env file configuration")
        st.stop()

def check_env_file():
    """Check if .env file exists and show setup instructions if not."""
    if not os.path.exists('.env'):
        st.error(".env file not found!")
        st.stop()

def find_csv_file():
    """Find the financial statements CSV file in the current directory."""
    patterns = [
        "financial_25 1 2.csv",
        "financial_statements_metadata_*.csv",
        "financial_*.csv"
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return files[0]
    
    all_csv_files = glob.glob("*.csv")
    if all_csv_files:
        st.error(f"Expected CSV file not found. Available CSV files: {', '.join(all_csv_files)}")
    else:
        st.error("No CSV files found in the current directory.")
    
    return None

def main():
    st.set_page_config(
        page_title="Financial Statement Extraction",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Financial Statement Extraction & Upload")
    
    check_env_file()
    app = get_app()
    
    # Sidebar for cache management
    with st.sidebar:
        st.header("Cache Management")
        
        if st.button("Show Cache Stats", key="show_cache_stats"):
            app.show_cache_stats()
        
        if st.button("Clear Cache", key="clear_cache", help="Clear all cached analysis results"):
            if st.session_state.get('confirm_clear_cache', False):
                app.clear_metadata_cache()
                st.session_state.confirm_clear_cache = False
            else:
                st.session_state.confirm_clear_cache = True
                st.warning("Click again to confirm cache clearing")
        
        # Reset confirmation if user does something else
        if 'confirm_clear_cache' in st.session_state and st.session_state.confirm_clear_cache:
            st.info("Cache clearing ready - click 'Clear Cache' again to confirm")
            
        st.write("---")
        st.header("AWS Connection")
        
        if st.button("Test AWS Connection", key="test_aws"):
            app.test_aws_connection()
        
        st.write("**AWS Info:**")
        st.write("- Tests S3 bucket access")
        st.write("- Verifies write permissions")
        st.write("- Shows available buckets")
        st.write("---")
        st.write("**Cache Info:**")
        st.write("- Speeds up repeated analyses")
        st.write("- Stores page locations for each statement")
        st.write("- Validates PDF content changes")
        
    # Load CSV file
    csv_file_path = find_csv_file()
    if not csv_file_path:
        st.stop()
    
    try:
        df = pd.read_csv(csv_file_path)
        st.success(f"Loaded {len(df)} records from {csv_file_path}")
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        st.stop()

    # Add debug section right after loading CSV
    if st.button("🔍 Debug Cache", key="debug_cache"):
        st.write("**Cache Debug Info:**")
        st.write(f"Cache file exists: {os.path.exists(app.cache_file)}")
        st.write(f"Cache file path: {os.path.abspath(app.cache_file)}")
        
        if os.path.exists(app.cache_file):
            cache_df = pd.read_csv(app.cache_file)
            st.write(f"Cache entries: {len(cache_df)}")
            if len(cache_df) > 0:
                st.dataframe(cache_df.tail())

    # Helper function for safe sorting
    def safe_sort_unique(series):
        unique_vals = series.dropna().unique()
        string_vals = [str(val) for val in unique_vals]
        return sorted(string_vals)

    # Filters for table extraction
    st.header("Extraction Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbols = [''] + safe_sort_unique(df['symbol'])
        selected_symbol = st.selectbox("Symbol", symbols, key="symbol_filter")
        
        statement_types = [''] + safe_sort_unique(df['statement_type'])
        selected_statement_type = st.selectbox("Statement Type", statement_types, key="statement_type_filter")

    with col2:
        # Filter period end dates based on selected symbol
        if selected_symbol:
            symbol_filtered_df = df[df['symbol'].astype(str) == str(selected_symbol)]
            period_end_dates = [''] + safe_sort_unique(symbol_filtered_df['period_end_date'])
        else:
            period_end_dates = ['']
        
        selected_period_end_date = st.selectbox("Period End Date", period_end_dates, key="period_end_date_filter")
        
        # Filter report types based on selected symbol and period end date
        if selected_symbol and selected_period_end_date:
            filtered_for_report_types = df[
                (df['symbol'].astype(str) == str(selected_symbol)) &
                (df['period_end_date'].astype(str) == str(selected_period_end_date))
            ]
            report_types = [''] + safe_sort_unique(filtered_for_report_types['report_type'])
        elif selected_symbol:
            symbol_filtered_df = df[df['symbol'].astype(str) == str(selected_symbol)]
            report_types = [''] + safe_sort_unique(symbol_filtered_df['report_type'])
        else:
            report_types = ['']
            
        selected_report_type = st.selectbox("Report Type", report_types, key="report_type_filter")

    with col3:
        # Page number will be populated after PDF analysis
        st.write("Page Number")
        page_placeholder = st.empty()
        
    # Find matching records
    if selected_symbol and selected_statement_type and selected_period_end_date and selected_report_type:
        
        # Fix the data labeling issue - the CSV columns are swapped
        actual_statement_type = selected_report_type  # This contains the real statement type (balance_sheet, cashflow, etc.)
        actual_report_type = selected_statement_type  # This contains the audit status (audited, unaudited)
        
        # Filter dataframe
        filtered_df = df[
            (df['symbol'].astype(str) == str(selected_symbol)) &
            (df['statement_type'].astype(str) == str(selected_statement_type)) &
            (df['period_end_date'].astype(str) == str(selected_period_end_date)) &
            (df['report_type'].astype(str) == str(selected_report_type))
        ]
        
        if len(filtered_df) > 0:
            record = filtered_df.iloc[0]
            pdf_path = record['pdf_s3_path']
            
            if pd.notna(pdf_path) and pdf_path.strip():
                st.success(f"Found matching record. Loading PDF...")
                
                # Download PDF
                pdf_content = app.download_from_s3(pdf_path.strip())
                
                if pdf_content:
                    # Get PDF page count
                    try:
                        doc = fitz.open(stream=pdf_content, filetype="pdf")
                        num_pages = len(doc)
                        doc.close()
                        
                        st.info(f"PDF loaded successfully ({num_pages} pages)")
                        
                        # Check if we should display extracted data results
                        if app.display_extracted_data_from_content_page():
                            # Data is being displayed, don't show other UI elements
                            pass
                        # Check if we should handle statement extraction
                        elif app.handle_statement_extraction(pdf_content):
                            # Extraction is being handled, don't show other UI elements
                            pass
                        # Check if we should show content page
                        elif st.button("📋 Generate Content Overview", type="primary", key="generate_content_overview"):
                            with st.spinner("Analyzing PDF for all statement types..."):
                                all_statements = app.analyze_all_statements_in_pdf(pdf_content, selected_symbol, selected_period_end_date)
                                if all_statements:
                                    st.session_state.all_statements_found = all_statements
                                    st.session_state.show_content_page = True
                                    st.rerun()
                        # Show content page if it exists in session state
                        elif st.session_state.get('show_content_page', False) and 'all_statements_found' in st.session_state:
                            app.display_content_page(
                                pdf_content, 
                                st.session_state.all_statements_found, 
                                selected_symbol, 
                                selected_period_end_date, 
                                selected_report_type
                            )
                            
                            # Add button to go back to individual extraction mode
                            if st.button("🔧 Switch to Individual Extraction Mode", key="switch_to_individual"):
                                st.session_state.show_content_page = False
                                if 'all_statements_found' in st.session_state:
                                    del st.session_state.all_statements_found
                                st.rerun()
                        else:
                            # Original individual extraction interface
                            st.subheader("Individual Statement Extraction")
                            
                            # Show statement type aliases being used
                            consolidated_terms, standard_terms = app.get_statement_search_terms(actual_statement_type)
                            if consolidated_terms or standard_terms:
                                with st.expander("Statement Search Terms (Aliases)", expanded=False):
                                    if consolidated_terms:
                                        st.write("**Priority 1 - Consolidated/Group terms:**")
                                        for term in consolidated_terms:
                                            st.write(f"  • {term}")
                                    if standard_terms:
                                        st.write("**Priority 2 - Standard terms:**")
                                        for term in standard_terms:
                                            st.write(f"  • {term}")
                        
                            # Find the correct page using Claude with multiple methods
                            if not st.session_state.get('show_content_page', False):
                                col_a, col_b = st.columns(2)
                            
                                with col_a:
                                    if st.button("Find Statement Page (Enhanced)", type="primary"):
                                        # Clear any previous results
                                        if 'detailed_analysis_result' in st.session_state:
                                            del st.session_state.detailed_analysis_result
                                        
                                        with st.spinner("Analyzing PDF with enhanced method..."):
                                            page_info = app.find_statement_page_robust_with_cache(
                                                pdf_content,
                                                actual_statement_type,
                                                actual_report_type,
                                                pdf_path.strip(),
                                                selected_symbol,
                                                selected_period_end_date
                                            )
                                        
                                        if page_info:
                                            # Show cache status
                                            if page_info.get('from_cache', False):
                                                st.success("✅ Result loaded from cache!")
                                                st.info(f"Cache timestamp: {page_info.get('cache_timestamp', 'Unknown')}")
                                            else:
                                                st.success("🔍 Fresh analysis completed and cached!")
                                            
                                            # Handle both single and multi-page results
                                            page_positions = page_info.get('pdf_page_positions', [])
                                            if not page_positions:
                                                # Fallback to single page format
                                                single_page = page_info.get('pdf_page_position') or page_info.get('page_number', 1)
                                                page_positions = [single_page]
                                            
                                            is_multi_page = page_info.get('is_multi_page', False)
                                            pages_text = ", ".join([str(p) for p in page_positions])
                                            printed_pages = page_info.get('printed_page_numbers', [])
                                            printed_text = ", ".join([str(p) for p in printed_pages]) if printed_pages else "Unknown"
                                            statement_priority = page_info.get('statement_priority', 'unknown')
                                            
                                            st.success(f"Found statement on PDF page(s): {pages_text}")
                                            st.info(f"Printed page numbers: {printed_text}")
                                            if statement_priority == 'consolidated':
                                                st.success(f"✅ Found CONSOLIDATED/GROUP statement (highest priority)")
                                            elif statement_priority == 'standard':
                                                st.info(f"📄 Found STANDARD statement (fallback)")
                                            if is_multi_page:
                                                st.info("Multi-page statement detected")
                                            st.info(f"Statement title: {page_info.get('statement_title', 'Unknown')}")
                                            st.info(f"Confidence: {page_info.get('confidence', 'Unknown')}")
                                            st.info(f"Reasoning: {page_info.get('reasoning', 'None provided')}")
                                            
                                            # Store in session state
                                            st.session_state.found_pages = page_positions
                                            st.session_state.is_multi_page = is_multi_page
                                            st.session_state.statement_title = page_info.get('statement_title', actual_statement_type)
                                            st.session_state.statement_priority = statement_priority
                                            st.session_state.pdf_content = pdf_content
                                        elif not st.session_state.get('detailed_analysis_result'):
                                            st.info("Enhanced method may show candidate options above. If multiple options appear, select the appropriate one.")
                                
                                with col_b:
                                    if st.button("Find Statement Page (Original)", type="secondary"):
                                        with st.spinner("Analyzing PDF with original method..."):
                                            page_info = app.find_statement_page(
                                                pdf_content,
                                                actual_statement_type,
                                                actual_report_type
                                            )
                                        
                                        if page_info:
                                            # Handle backward compatibility - convert single page to list
                                            page_number = page_info.get('pdf_page_position') or page_info.get('page_number', 1)
                                            page_positions = [page_number]
                                            statement_priority = page_info.get('statement_priority', 'unknown')
                                            
                                            st.success(f"Found statement on PDF page {page_number}")
                                            st.info(f"Statement title: {page_info.get('statement_title', 'Unknown')}")
                                            st.info(f"Printed page: {page_info.get('printed_page_number', 'Unknown')}")
                                            if statement_priority == 'consolidated':
                                                st.success(f"✅ Found CONSOLIDATED/GROUP statement (highest priority)")
                                            elif statement_priority == 'standard':
                                                st.info(f"📄 Found STANDARD statement (fallback)")
                                            st.info(f"Confidence: {page_info.get('confidence', 'Unknown')}")
                                            st.info(f"Reasoning: {page_info.get('reasoning', 'None provided')}")
                                            
                                            # Store in session state
                                            st.session_state.found_pages = page_positions
                                            st.session_state.is_multi_page = False
                                            st.session_state.statement_title = page_info.get('statement_title', actual_statement_type)
                                            st.session_state.statement_priority = statement_priority
                                            st.session_state.pdf_content = pdf_content
                                        else:
                                            st.error("Could not find the statement page with original method")
                                
                                # Page selection dropdown
                                if 'found_pages' in st.session_state:
                                    found_pages = st.session_state.found_pages
                                    is_multi_page = st.session_state.get('is_multi_page', False)
                                    statement_priority = st.session_state.get('statement_priority', 'unknown')
                                    
                                    # Show priority information
                                    if statement_priority == 'consolidated':
                                        st.success("🎯 Using CONSOLIDATED/GROUP statement (highest priority)")
                                    elif statement_priority == 'standard':
                                        st.info("📋 Using STANDARD statement (consolidated/group not found)")
                                    
                                    if is_multi_page and len(found_pages) > 1:
                                        # Multi-page statement
                                        st.subheader("Multi-Page Statement Detected")
                                        pages_text = ", ".join([str(p) for p in found_pages])
                                        st.info(f"Statement spans pages: {pages_text}")
                                        
                                        # Allow user to modify the page selection
                                        st.write("Select pages to extract:")
                                        page_options = list(range(1, num_pages + 1))
                                        
                                        selected_pages = st.multiselect(
                                            "Pages to extract:",
                                            page_options,
                                            default=found_pages,
                                            key="multi_page_selector"
                                        )
                                    else:
                                        # Single page statement
                                        page_options = list(range(1, num_pages + 1))
                                        default_page_idx = page_options.index(found_pages[0]) if found_pages[0] in page_options else 0
                                        
                                        selected_page = st.selectbox(
                                            "Select Page to Extract",
                                            page_options,
                                            index=default_page_idx,
                                            key="single_page_selector"
                                        )
                                        selected_pages = [selected_page]
                                else:
                                    # Manual page selection when no auto-detection was run
                                    st.write("Manual Page Selection")
                                    page_options = list(range(1, num_pages + 1))
                                    
                                    # Allow user to choose between single or multiple pages
                                    selection_mode = st.radio(
                                        "Selection mode:",
                                        ["Single page", "Multiple pages"],
                                        key="selection_mode"
                                    )
                                    
                                    if selection_mode == "Single page":
                                        selected_page = st.selectbox(
                                            "Select Page to Extract", 
                                            page_options,
                                            key="manual_single_page_selector"
                                        )
                                        selected_pages = [selected_page]
                                    else:
                                        selected_pages = st.multiselect(
                                            "Select Pages to Extract and Merge:",
                                            page_options,
                                            default=[1],
                                            key="manual_multi_page_selector",
                                            help="Select multiple pages if the statement spans across pages"
                                        )
                                        if not selected_pages:
                                            st.warning("Please select at least one page")
                                            selected_pages = [1]
                                
                                # Display selected pages
                                if selected_pages:
                                    if len(selected_pages) == 1:
                                        st.subheader(f"Page {selected_pages[0]} Preview")
                                        img_data = app.render_pdf_page(pdf_content, selected_pages[0] - 1)
                                        if img_data:
                                            st.image(img_data, caption=f"Page {selected_pages[0]}", use_column_width=True)
                                    else:
                                        st.subheader(f"Pages {', '.join(map(str, selected_pages))} Preview")
                                        for page_num in selected_pages[:3]:  # Show max 3 page previews
                                            img_data = app.render_pdf_page(pdf_content, page_num - 1)
                                            if img_data:
                                                st.image(img_data, caption=f"Page {page_num}", use_column_width=True)
                                        if len(selected_pages) > 3:
                                            st.info(f"... and {len(selected_pages) - 3} more page(s)")
                                    
                                    # Extraction methods
                                    st.subheader("Extract Data")
                                    
                                    extraction_method = st.radio(
                                        "Choose extraction method:",
                                        ["Claude Sonnet 4 (Vertex AI)", "AWS Textract"]
                                    )
                                    
                                    # Separate Extract and Upload buttons with verification step
                                    col_extract, col_upload = st.columns(2)
                                    
                                    with col_extract:
                                        if st.button("Extract Financial Data", type="primary"):
                                            # Get the correct statement type and report type
                                            actual_statement_type = selected_report_type  # This contains the real statement type
                                            actual_report_type = selected_statement_type  # This contains the audit status
                                            
                                            # Extract data only
                                            success, csv_content = app.extract_statement_data_only(
                                                pdf_content=pdf_content,
                                                selected_pages=selected_pages,
                                                statement_type=actual_statement_type,
                                                report_type=actual_report_type,
                                                extraction_method=extraction_method
                                            )
                                            
                                            if success:
                                                # Store extracted data for upload
                                                st.session_state.extracted_data = {
                                                    'csv_content': csv_content,
                                                    'symbol': selected_symbol,
                                                    'statement_type': actual_statement_type,
                                                    'period_end_date': selected_period_end_date,
                                                    'report_type': actual_report_type,
                                                    'statement_title': st.session_state.get('statement_title', ''),
                                                    'pdf_path': pdf_path
                                                }
                                                
                                                # Display extracted data
                                                st.subheader("Extracted CSV Data")
                                                try:
                                                    extracted_df = pd.read_csv(io.StringIO(csv_content))
                                                    st.dataframe(extracted_df, use_container_width=True)
                                                    st.info(f"Extracted {len(extracted_df)} rows of data")
                                                except Exception as e:
                                                    st.text_area("Raw CSV Data:", csv_content, height=400)
                                                    st.warning("Could not parse as CSV - showing raw data above")
                                    
                                    with col_upload:
                                        # Upload button (only show if data is extracted)
                                        if 'extracted_data' in st.session_state:
                                            # First, show verification step if not already verified
                                            if not st.session_state.get('upload_verified', False):
                                                if st.button("📋 Review Upload Details", type="secondary"):
                                                    st.session_state.show_upload_verification = True
                                                    st.rerun()
                                            else:
                                                # Show actual upload button after verification
                                                if st.button("Upload to S3", type="primary"):
                                                    extracted_info = st.session_state.extracted_data
                                                    
                                                    with st.spinner("Uploading to S3..."):
                                                        # Use the verified filename and path
                                                        verified_s3_path = st.session_state.get('verified_s3_path', '')
                                                        
                                                        # Upload using the custom filename and path
                                                        success = app.upload_to_s3_with_custom_path(
                                                            csv_content=extracted_info['csv_content'],
                                                            custom_s3_path=verified_s3_path
                                                        )
                                                    
                                                    if success:
                                                        st.success("✅ Upload completed successfully!")
                                                        
                                                        # Update spreadsheet with verified path
                                                        new_row_data = {
                                                            'symbol': extracted_info['symbol'],
                                                            'statement_type': selected_statement_type,
                                                            'period': record.get('period', ''),
                                                            'period_detail': record.get('period_detail', ''),
                                                            'period_end_date': extracted_info['period_end_date'],
                                                            'report_type': selected_report_type,
                                                            'consolidation_type': record.get('consolidation_type', ''),
                                                            'status': 1.0,
                                                            's3_path': verified_s3_path,
                                                            'pdf_folder_path': record.get('pdf_folder_path', ''),
                                                            'statement_ID': f"{extracted_info['symbol']}_{selected_statement_type}_{extracted_info['period_end_date']}",
                                                            'pdf_s3_path': extracted_info['pdf_path'],
                                                            'is_missing': False,
                                                            'is_mismatched': False,
                                                            'is_incomplete': False,
                                                            'is_incomplete_note': '',
                                                            'is_mismatched_note': '',
                                                            'updated': True
                                                        }
                                                        
                                                        spreadsheet_id = "1KRZS4EAo7Rq-7ISmkG1QrdmgnAOGY5PCWoa5ApRLQ8g"
                                                        tab_name = "financial_statements_metadata_3_09_2025 - financial_statements_metadata_3_09_2025 (1)"
                                                        
                                                        with st.spinner("Updating spreadsheet..."):
                                                            if app.add_row_to_spreadsheet(spreadsheet_id, tab_name, new_row_data):
                                                                st.success("📊 Spreadsheet updated successfully!")
                                                            else:
                                                                st.warning("⚠️ File uploaded but failed to update spreadsheet")
                                                        
                                                        # Clear session state after successful upload
                                                        for key in ['extracted_data', 'upload_verified', 'verified_filename', 'verified_s3_path', 'show_upload_verification']:
                                                            if key in st.session_state:
                                                                del st.session_state[key]
                                                    else:
                                                        st.error("❌ Upload failed")
                                        else:
                                            st.info("Extract data first to enable upload")
                            
                            # Add the verification dialog after the extraction section
                            if st.session_state.get('show_upload_verification', False) and 'extracted_data' in st.session_state:
                                st.markdown("---")
                                st.subheader("📋 Upload Verification")
                                
                                extracted_info = st.session_state.extracted_data
                                
                                # Generate initial filename and S3 path
                                statement_title = extracted_info.get('statement_title', extracted_info['statement_type'])
                                clean_title = statement_title.lower().replace(' ', '_').replace('/', '_').replace('&', 'and')
                                
                                # Auto-generate filename
                                auto_filename = f"{extracted_info['symbol'].lower()}-{clean_title}-{app.format_period_end_date(extracted_info['period_end_date'])}.csv"
                                
                                # Auto-generate S3 path
                                auto_s3_path = app.generate_s3_path(
                                    symbol=extracted_info['symbol'],
                                    statement_title=clean_title,
                                    period_end_date=extracted_info['period_end_date'],
                                    report_type=extracted_info['report_type'],
                                    statement_type=extracted_info['report_type']  # Pass audit status
                                )
                                
                                st.info("Review and modify the upload details before proceeding:")
                                
                                # Editable filename
                                verified_filename = st.text_input(
                                    "CSV Filename:",
                                    value=st.session_state.get('verified_filename', auto_filename),
                                    key="filename_verification",
                                    help="Modify the filename if needed"
                                )
                                
                                # Editable S3 path
                                verified_s3_path = st.text_area(
                                    "S3 Upload Path:",
                                    value=st.session_state.get('verified_s3_path', auto_s3_path),
                                    key="s3_path_verification",
                                    height=100,
                                    help="Modify the S3 path if needed. Must start with 's3://'"
                                )
                                
                                # Validation
                                path_valid = True
                                if not verified_s3_path.startswith('s3://'):
                                    st.error("S3 path must start with 's3://'")
                                    path_valid = False
                                
                                if not verified_filename.endswith('.csv'):
                                    st.warning("Filename should end with '.csv'")
                                
                                # Show what will be uploaded
                                col_info1, col_info2 = st.columns(2)
                                
                                with col_info1:
                                    st.write("**Upload Summary:**")
                                    st.write(f"• Symbol: {extracted_info['symbol']}")
                                    st.write(f"• Statement: {extracted_info['statement_type']}")
                                    st.write(f"• Period: {extracted_info['period_end_date']}")
                                    st.write(f"• Report Type: {extracted_info['report_type']}")
                                
                                with col_info2:
                                    st.write("**File Details:**")
                                    try:
                                        csv_df = pd.read_csv(io.StringIO(extracted_info['csv_content']))
                                        st.write(f"• Rows: {len(csv_df)}")
                                        st.write(f"• Columns: {len(csv_df.columns)}")
                                        st.write(f"• Size: ~{len(extracted_info['csv_content'])} chars")
                                    except:
                                        st.write(f"• Size: ~{len(extracted_info['csv_content'])} chars")
                                
                                # Buttons
                                col_confirm, col_cancel, col_download = st.columns(3)
                                
                                with col_confirm:
                                    if st.button("✅ Confirm & Continue", type="primary", disabled=not path_valid):
                                        # Store verified details
                                        st.session_state.verified_filename = verified_filename
                                        st.session_state.verified_s3_path = verified_s3_path
                                        st.session_state.upload_verified = True
                                        st.session_state.show_upload_verification = False
                                        st.success("Upload details verified! You can now upload to S3.")
                                        st.rerun()
                                
                                with col_cancel:
                                    if st.button("❌ Cancel", type="secondary"):
                                        st.session_state.show_upload_verification = False
                                        st.rerun()
                                
                                with col_download:
                                    # Download option with verified filename
                                    st.download_button(
                                        label="💾 Download CSV",
                                        data=extracted_info['csv_content'],
                                        file_name=verified_filename,
                                        mime="text/csv",
                                        key="download_verified_csv"
                                    )
                        
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")
                else:
                    st.error("Could not download PDF file")
            else:
                st.error("No PDF path found for this record")
        else:
            st.warning("No matching records found with the selected filters")
    else:
        st.info("Please select all filters to proceed with extraction")

if __name__ == "__main__":
    main()