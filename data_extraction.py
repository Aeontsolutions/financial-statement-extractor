"""
Data Extraction Module for Financial Statement Processing

This module contains the core data extraction methods and supporting functions
extracted from the main Financial Statement Extraction app.
"""

import base64
import hashlib
import json
import os
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests
import pandas as pd
import boto3
import fitz  # PyMuPDF
from botocore.exceptions import ClientError


class DataExtractor:
    """
    Core data extraction functionality for financial statements.
    Supports both Claude AI and AWS Textract extraction methods.
    """
    
    def __init__(self, aws_access_key_id: str, aws_secret_access_key: str, 
                 aws_region: str = 'us-east-1', project_id: str = None, 
                 vertex_region: str = 'us-east5', credentials=None,
                 cache_file: str = "pdf_analysis_cache.csv"):
        """
        Initialize the data extractor with necessary credentials.
        
        Args:
            aws_access_key_id: AWS access key
            aws_secret_access_key: AWS secret key
            aws_region: AWS region for services
            project_id: Google Cloud project ID for Vertex AI
            vertex_region: Vertex AI region
            credentials: Google Cloud credentials object
            cache_file: Path to the cache file for storing analysis results
        """
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region = aws_region
        self.project_id = project_id
        self.vertex_region = vertex_region
        self.credentials = credentials
        
        # Setup caching
        self.cache_file = cache_file
        self.cache_columns = [
            'pdf_s3_path', 'symbol', 'period_end_date', 'report_type', 'statement_type',
            'pdf_page_positions', 'printed_page_numbers', 'statement_title', 
            'statement_priority', 'is_multi_page', 'confidence', 'reasoning',
            'analysis_timestamp', 'pdf_hash'
        ]
        self.initialize_cache_file()
        
        # Initialize AWS clients
        self.setup_aws_clients()
        
        # Setup Vertex AI endpoint
        if project_id and vertex_region:
            self.vertex_endpoint = f"https://{vertex_region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{vertex_region}/publishers/anthropic/models/claude-3-5-sonnet-v2@20241022:streamRawPredict"
        
        # Setup statement aliases
        self.setup_statement_aliases()
    
    def initialize_cache_file(self):
        """Initialize the cache file if it doesn't exist."""
        try:
            if not os.path.exists(self.cache_file):
                df = pd.DataFrame(columns=self.cache_columns)
                df.to_csv(self.cache_file, index=False)
                print(f"Initialized cache file: {self.cache_file}")
        except Exception as e:
            print(f"Could not initialize cache file: {e}")
    
    def generate_pdf_hash(self, pdf_content: bytes) -> str:
        """Generate a hash of the PDF content for cache validation."""
        try:
            return hashlib.md5(pdf_content).hexdigest()[:16]  # Use first 16 chars for brevity
        except Exception as e:
            print(f"Could not generate PDF hash: {e}")
            return "unknown"
    
    def check_metadata_cache(self, pdf_s3_path: str, symbol: str, period_end_date: str, 
                           report_type: str, statement_type: str, pdf_content: bytes) -> Optional[Dict]:
        """Check if analysis results exist in cache."""
        try:
            if not os.path.exists(self.cache_file):
                return None
            
            # Load cache
            cache_df = pd.read_csv(self.cache_file)
            
            if cache_df.empty:
                return None
            
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
            
            if not matching_entries.empty:
                entry = matching_entries.iloc[0]
                cached_hash = entry.get('pdf_hash', 'unknown')
                
                # Validate PDF hasn't changed
                if cached_hash != 'unknown' and cached_hash != current_pdf_hash:
                    print(f"PDF content has changed. Cache invalid for {symbol}_{statement_type}")
                    return None
                
                # Parse cached result
                try:
                    pdf_pages = eval(entry['pdf_page_positions']) if isinstance(entry['pdf_page_positions'], str) else entry['pdf_page_positions']
                    printed_pages = eval(entry['printed_page_numbers']) if isinstance(entry['printed_page_numbers'], str) else entry['printed_page_numbers']
                    
                    result = {
                        'pdf_page_positions': pdf_pages,
                        'printed_page_numbers': printed_pages,
                        'statement_title': entry['statement_title'],
                        'statement_priority': entry['statement_priority'],
                        'is_multi_page': entry['is_multi_page'],
                        'confidence': entry['confidence'],
                        'reasoning': entry['reasoning']
                    }
                    
                    print(f"Found cached result for {symbol}_{statement_type}: pages {pdf_pages}")
                    return result
                    
                except Exception as parse_error:
                    print(f"Error parsing cached entry: {parse_error}")
                    return None
            
            return None
            
        except Exception as e:
            print(f"Error checking metadata cache: {e}")
            return None
    
    def save_to_metadata_cache(self, pdf_s3_path: str, symbol: str, period_end_date: str, 
                             report_type: str, statement_type: str, analysis_result: Dict, pdf_content: bytes):
        """Save analysis results to cache."""
        try:
            # Generate PDF hash
            pdf_hash = self.generate_pdf_hash(pdf_content)
            
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
            print(f"Saved analysis result to cache: {self.cache_file}")
            
        except Exception as e:
            print(f"Could not save to metadata cache: {e}")
    
    def find_statement_page_robust(self, pdf_content: bytes, statement_type: str, report_type: str) -> Dict:
        """
        Enhanced page finding method with better handling of multi-page statements.
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
                prompt += "\n2. FALLBACK - Standard statements (use only if consolidated/group not found):\n"
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
            if not access_token:
                print("Error: No valid access token available for Claude API")
                return None
            
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
                result = response.json()
                if 'content' in result and len(result['content']) > 0:
                    response_text = result['content'][0]['text'].strip()
                    try:
                        # Try to parse JSON response
                        parsed_result = json.loads(response_text)
                        return parsed_result
                    except json.JSONDecodeError:
                        print(f"Could not parse JSON from Claude response: {response_text}")
                        return None
            else:
                print(f"Claude API error: {response.status_code} - {response.text}")
                return None
            
            return None
                
        except Exception as e:
            print(f"Error in robust page finding: {e}")
            return None

    def find_statement_page_robust_with_cache(self, pdf_content: bytes, statement_type: str, 
                                            report_type: str, pdf_s3_path: str = "", 
                                            symbol: str = "", period_end_date: str = "") -> Dict:
        """Enhanced page finding with caching support."""
        try:
            # Check cache first if we have the required metadata
            if pdf_s3_path and symbol and period_end_date:
                cached_result = self.check_metadata_cache(
                    pdf_s3_path, symbol, period_end_date, report_type, statement_type, pdf_content
                )
                
                if cached_result:
                    print(f"Using cached result for {symbol}_{statement_type}")
                    return cached_result
            
            # Run normal analysis if not in cache
            print("No cached result found. Running fresh analysis...")
            result = self.find_statement_page_robust(pdf_content, statement_type, report_type)
            
            if result and pdf_s3_path and symbol and period_end_date:
                # Save to cache
                self.save_to_metadata_cache(
                    pdf_s3_path, symbol, period_end_date, report_type, statement_type, result, pdf_content
                )
                
            return result
            
        except Exception as e:
            print(f"Error in cached page finding: {e}")
            return None
    
    def analyze_all_statements_in_pdf(self, pdf_content: bytes, symbol: str = "", 
                                    period_end_date: str = "", pdf_s3_path: str = "") -> Dict:
        """Analyze PDF to find all available statement types and their pages."""
        try:
            # Get total page count
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            total_pages = len(doc)
            doc.close()
            
            print(f"Analyzing PDF for all statement types ({total_pages} pages)...")
            
            # Define all possible statement types to search for
            statement_types_to_search = ['balance_sheet', 'cashflow_statement', 'income_statement']
            
            all_statements_found = {}
            
            for stmt_type in statement_types_to_search:
                print(f"Searching for {stmt_type}...")
                
                # Try both audited and unaudited as report types
                for report_type in ['audited', 'unaudited']:
                    result = self.find_statement_page_robust_with_cache(
                        pdf_content, stmt_type, report_type, pdf_s3_path, symbol, period_end_date
                    )
                    
                    if result and result.get('pdf_page_positions'):
                        print(f"Found {stmt_type} ({report_type}): pages {result['pdf_page_positions']}")
                        
                        # Store the result with all necessary info
                        all_statements_found[stmt_type] = {
                            'pages': result['pdf_page_positions'],
                            'title': result.get('statement_title', 'Unknown'),
                            'priority': result.get('statement_priority', 'standard'),
                            'confidence': result.get('confidence', 'medium'),
                            'reasoning': result.get('reasoning', ''),
                            'is_multi_page': result.get('is_multi_page', False),
                            'report_type': report_type
                        }
                        break  # Found one, no need to try other report types
                
                if stmt_type not in all_statements_found:
                    print(f"Could not locate {stmt_type} in PDF")
            
            return all_statements_found
            
        except Exception as e:
            print(f"Error analyzing PDF for all statements: {e}")
            return {}
    
    def setup_aws_clients(self):
        """Initialize AWS S3 and Textract clients."""
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
    
    def get_access_token(self) -> str:
        """Get access token for Vertex AI API."""
        if self.credentials and hasattr(self.credentials, 'valid'):
            if not self.credentials.valid:
                from google.auth.transport.requests import Request
                self.credentials.refresh(Request())
            return self.credentials.token
        return None
    
    def extract_with_claude(self, pdf_content: bytes, page_numbers: List[int], 
                          statement_type: str, report_type: str = None) -> Optional[str]:
        """
        Extract financial data from multiple pages using Claude and consolidate into one CSV.
        
        Args:
            pdf_content: PDF file content as bytes
            page_numbers: List of page numbers to extract from
            statement_type: Type of financial statement
            report_type: Type of report (optional)
            
        Returns:
            CSV string with extracted data or None if extraction fails
        """
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
            if not access_token:
                print("Error: No valid access token available for Claude API")
                return None
            
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
                result = response.json()
                if 'content' in result and len(result['content']) > 0:
                    return result['content'][0]['text'].strip()
            else:
                print(f"Claude API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error with Claude extraction: {e}")
            traceback.print_exc()
            return None
    
    def extract_with_textract(self, pdf_content: bytes, page_numbers: List[int], 
                            statement_type: str = None, report_type: str = None) -> Optional[str]:
        """
        Extract financial data from multiple pages using AWS Textract and consolidate into one CSV.
        
        Args:
            pdf_content: PDF file content as bytes
            page_numbers: List of page numbers to extract from
            statement_type: Type of financial statement (optional)
            report_type: Type of report (optional)
            
        Returns:
            CSV string with extracted data or None if extraction fails
        """
        try:
            if isinstance(page_numbers, int):
                page_numbers = [page_numbers]
            
            # Display what we're extracting
            pages_text = ", ".join([str(p) for p in page_numbers])
            print(f"Statement Type: {statement_type or 'Unknown Statement Type'}")
            print(f"Report Type: {report_type or 'Not specified'}")
            print(f"Pages: {pages_text}")
            print(f"Method: AWS Textract")
            print("Note: Textract extracts table structure but cannot validate statement type.")
            
            all_tables = []
            
            for page_number in page_numbers:
                print(f"Processing page {page_number} with Textract...")
                
                # Extract specific page from PDF
                page_pdf = self.extract_pdf_chunk(pdf_content, page_number, page_number)
                if not page_pdf:
                    print(f"Failed to extract page {page_number}")
                    continue
                
                try:
                    # Call Textract
                    response = self.textract_client.analyze_document(
                        Document={'Bytes': page_pdf},
                        FeatureTypes=['TABLES']
                    )
                    
                    # Extract tables from response
                    page_tables = self.extract_tables_from_blocks(response['Blocks'])
                    
                    if page_tables:
                        print(f"Found {len(page_tables)} tables on page {page_number}")
                        for i, table in enumerate(page_tables):
                            all_tables.append((page_number, table))
                    else:
                        print(f"No tables found on page {page_number}")
                        
                except ClientError as e:
                    print(f"Textract error for page {page_number}: {e}")
                    continue
                except Exception as e:
                    print(f"Unexpected error processing page {page_number}: {e}")
                    continue
            
            if not all_tables:
                print("No tables found in any of the specified pages")
                return None
            
            # Consolidate all tables into one CSV
            result_csv = self.consolidate_tables_to_csv(all_tables)
            
            if result_csv:
                print(f"Successfully consolidated {len(all_tables)} tables into CSV")
                return result_csv
            else:
                print("Failed to consolidate tables into CSV")
                return None
                
        except Exception as e:
            print(f"Error with Textract extraction: {e}")
            traceback.print_exc()
            return None
    
    def extract_pdf_chunk(self, pdf_content: bytes, start_page: int, end_page: int) -> Optional[bytes]:
        """
        Extract a chunk of pages from the PDF.
        
        Args:
            pdf_content: Original PDF content
            start_page: Starting page number (1-indexed)
            end_page: Ending page number (1-indexed)
            
        Returns:
            PDF bytes containing only the specified pages
        """
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
            print(f"Error extracting PDF chunk: {e}")
            return None
    
    def consolidate_tables_to_csv(self, page_tables: List[Tuple[int, List[List[str]]]]) -> str:
        """
        Consolidate multiple page tables into a single CSV.
        
        Args:
            page_tables: List of tuples containing (page_number, table_data)
            
        Returns:
            CSV string with consolidated data
        """
        try:
            consolidated_rows = []
            headers_added = False
            
            for page_num, table_data in page_tables:
                print(f"Consolidating table from page {page_num} with {len(table_data)} rows")
                
                if not table_data:
                    continue
                
                # If this is the first table, use its headers
                if not headers_added and table_data:
                    consolidated_rows.append(table_data[0])  # Header row
                    headers_added = True
                    if len(table_data) > 1:
                        consolidated_rows.extend(table_data[1:])  # Data rows
                else:
                    # For subsequent tables, skip headers and add only data rows
                    if len(table_data) > 1:
                        consolidated_rows.extend(table_data[1:])
            
            if not consolidated_rows:
                return ""
            
            # Convert to CSV format
            csv_content = self.table_to_csv(consolidated_rows)
            return csv_content
            
        except Exception as e:
            print(f"Error consolidating tables: {e}")
            traceback.print_exc()
            return ""
    
    def extract_tables_from_blocks(self, blocks):
        """
        Extract table data from Textract blocks.
        
        Args:
            blocks: Textract response blocks
            
        Returns:
            List of tables, each table is a list of rows
        """
        try:
            # Create a dictionary to map block IDs to blocks
            block_map = {block['Id']: block for block in blocks}
            
            # Find all table blocks
            table_blocks = [block for block in blocks if block['BlockType'] == 'TABLE']
            
            tables = []
            
            for table_block in table_blocks:
                # Get table cells
                table_data = []
                max_row = 0
                max_col = 0
                
                # Find the maximum row and column indices
                if 'Relationships' in table_block:
                    for relationship in table_block['Relationships']:
                        if relationship['Type'] == 'CHILD':
                            for cell_id in relationship['Ids']:
                                if cell_id in block_map:
                                    cell_block = block_map[cell_id]
                                    if cell_block['BlockType'] == 'CELL':
                                        row_index = cell_block['RowIndex'] - 1
                                        col_index = cell_block['ColumnIndex'] - 1
                                        max_row = max(max_row, row_index)
                                        max_col = max(max_col, col_index)
                
                # Initialize table grid
                table_grid = [[''] * (max_col + 1) for _ in range(max_row + 1)]
                
                # Fill the table grid
                if 'Relationships' in table_block:
                    for relationship in table_block['Relationships']:
                        if relationship['Type'] == 'CHILD':
                            for cell_id in relationship['Ids']:
                                if cell_id in block_map:
                                    cell_block = block_map[cell_id]
                                    if cell_block['BlockType'] == 'CELL':
                                        row_index = cell_block['RowIndex'] - 1
                                        col_index = cell_block['ColumnIndex'] - 1
                                        cell_text = self.get_cell_text(cell_block, block_map)
                                        table_grid[row_index][col_index] = cell_text
                
                if table_grid:
                    tables.append(table_grid)
            
            return tables
            
        except Exception as e:
            print(f"Error extracting tables from blocks: {e}")
            return []
    
    def get_cell_text(self, cell_block, block_map):
        """
        Extract text from a cell block.
        
        Args:
            cell_block: Textract cell block
            block_map: Dictionary mapping block IDs to blocks
            
        Returns:
            Text content of the cell
        """
        try:
            text = ""
            if 'Relationships' in cell_block:
                for relationship in cell_block['Relationships']:
                    if relationship['Type'] == 'CHILD':
                        for word_id in relationship['Ids']:
                            if word_id in block_map:
                                word_block = block_map[word_id]
                                if word_block['BlockType'] == 'WORD':
                                    text += word_block['Text'] + " "
            return text.strip()
            
        except Exception as e:
            print(f"Error getting cell text: {e}")
            return ""
    
    def table_to_csv(self, table_data):
        """
        Convert table data to CSV format.
        
        Args:
            table_data: List of rows, each row is a list of cell values
            
        Returns:
            CSV formatted string
        """
        try:
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            for row in table_data:
                # Clean up cell values
                cleaned_row = [str(cell).strip() if cell else '' for cell in row]
                writer.writerow(cleaned_row)
            
            return output.getvalue()
            
        except Exception as e:
            print(f"Error converting table to CSV: {e}")
            return ""


# Utility functions for standalone use
def extract_statement_data(pdf_path: str, page_numbers: List[int], statement_type: str,
                         method: str = 'claude', aws_credentials: dict = None,
                         google_credentials=None) -> Optional[str]:
    """
    Standalone function to extract financial statement data from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        page_numbers: List of page numbers to extract
        statement_type: Type of financial statement
        method: Extraction method ('claude' or 'textract')
        aws_credentials: Dict with AWS credentials
        google_credentials: Google Cloud credentials for Claude API
        
    Returns:
        CSV string with extracted data
    """
    try:
        # Read PDF file
        with open(pdf_path, 'rb') as f:
            pdf_content = f.read()
        
        # Initialize extractor
        extractor = DataExtractor(
            aws_access_key_id=aws_credentials.get('access_key_id'),
            aws_secret_access_key=aws_credentials.get('secret_access_key'),
            aws_region=aws_credentials.get('region', 'us-east-1'),
            project_id=aws_credentials.get('project_id'),
            vertex_region=aws_credentials.get('vertex_region', 'us-east5'),
            credentials=google_credentials
        )
        
        # Extract data
        if method.lower() == 'claude':
            return extractor.extract_with_claude(pdf_content, page_numbers, statement_type)
        elif method.lower() == 'textract':
            return extractor.extract_with_textract(pdf_content, page_numbers, statement_type)
        else:
            print(f"Unknown extraction method: {method}")
            return None
            
    except Exception as e:
        print(f"Error in standalone extraction: {e}")
        return None


def find_and_extract_statement(pdf_path: str, statement_type: str, method: str = 'claude',
                              aws_credentials: dict = None, google_credentials=None,
                              symbol: str = "", period_end_date: str = "") -> Optional[str]:
    """
    Complete workflow: Find statement pages automatically and then extract data.
    
    Args:
        pdf_path: Path to the PDF file
        statement_type: Type of financial statement to find and extract
        method: Extraction method ('claude' or 'textract')
        aws_credentials: Dict with AWS credentials
        google_credentials: Google Cloud credentials for Claude API
        symbol: Company symbol (for caching)
        period_end_date: Period end date (for caching)
        
    Returns:
        CSV string with extracted data
    """
    try:
        # Read PDF file
        with open(pdf_path, 'rb') as f:
            pdf_content = f.read()
        
        # Initialize extractor
        extractor = DataExtractor(
            aws_access_key_id=aws_credentials.get('access_key_id'),
            aws_secret_access_key=aws_credentials.get('secret_access_key'),
            aws_region=aws_credentials.get('region', 'us-east-1'),
            project_id=aws_credentials.get('project_id'),
            vertex_region=aws_credentials.get('vertex_region', 'us-east5'),
            credentials=google_credentials
        )
        
        # First, find the statement pages
        print(f"Looking for {statement_type} in PDF...")
        
        # Try both audited and unaudited
        for report_type in ['audited', 'unaudited']:
            page_result = extractor.find_statement_page_robust_with_cache(
                pdf_content, statement_type, report_type, 
                pdf_s3_path=pdf_path, symbol=symbol, period_end_date=period_end_date
            )
            
            if page_result and page_result.get('pdf_page_positions'):
                pages = page_result['pdf_page_positions']
                print(f"Found {statement_type} on pages: {pages}")
                
                # Now extract the data
                if method.lower() == 'claude':
                    return extractor.extract_with_claude(pdf_content, pages, statement_type, report_type)
                elif method.lower() == 'textract':
                    return extractor.extract_with_textract(pdf_content, pages, statement_type, report_type)
                else:
                    print(f"Unknown extraction method: {method}")
                    return None
        
        print(f"Could not find {statement_type} in the PDF")
        return None
            
    except Exception as e:
        print(f"Error in find and extract workflow: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    print("Data Extraction Module for Financial Statement Processing")
    print("This module provides Claude AI and AWS Textract extraction capabilities.")
    print("\nExample usage:")
    print("""
    from data_extraction import DataExtractor, find_and_extract_statement
    
    # Method 1: Manual page specification
    extractor = DataExtractor(
        aws_access_key_id='your_key',
        aws_secret_access_key='your_secret'
    )
    
    # Extract using Textract from known pages
    with open('financial_statement.pdf', 'rb') as f:
        pdf_content = f.read()
    
    csv_data = extractor.extract_with_textract(
        pdf_content, 
        page_numbers=[1, 2], 
        statement_type='balance_sheet'
    )
    
    # Method 2: Automatic page detection and extraction
    aws_creds = {
        'access_key_id': 'your_key',
        'secret_access_key': 'your_secret',
        'project_id': 'your_gcp_project',
        'vertex_region': 'us-east5'
    }
    
    csv_data = find_and_extract_statement(
        pdf_path='financial_statement.pdf',
        statement_type='balance_sheet',
        method='claude',
        aws_credentials=aws_creds,
        google_credentials=your_gcp_credentials,
        symbol='AAPL',
        period_end_date='2024-12-31'
    )
    """)