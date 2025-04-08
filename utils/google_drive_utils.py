"""
Utility functions for interacting with Google Drive API
"""
import os
import io
import re
import logging
import tempfile
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from urllib.parse import urlparse, parse_qs

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_folder_id_from_url(url):
    """
    Extract the folder ID from a Google Drive URL
    
    Args:
        url: Google Drive folder URL
        
    Returns:
        str: Folder ID or None if not found
    """
    # Handle different URL formats
    parsed_url = urlparse(url)
    
    # Format: https://drive.google.com/drive/folders/FOLDER_ID
    if 'drive.google.com/drive/folders' in url:
        path_parts = parsed_url.path.split('/')
        for i, part in enumerate(path_parts):
            if part == 'folders' and i + 1 < len(path_parts):
                return path_parts[i + 1]
    
    # Format: https://drive.google.com/open?id=FOLDER_ID
    elif 'drive.google.com/open' in url:
        query_params = parse_qs(parsed_url.query)
        if 'id' in query_params:
            return query_params['id'][0]
    
    # Format: https://drive.google.com/drive/u/0/folders/FOLDER_ID
    elif re.search(r'drive\.google\.com/drive/u/\d+/folders', url):
        path_parts = parsed_url.path.split('/')
        for i, part in enumerate(path_parts):
            if part == 'folders' and i + 1 < len(path_parts):
                return path_parts[i + 1]
    
    # Format: https://drive.google.com/drive/shared-with-me/FOLDER_ID
    elif 'drive.google.com/drive/shared-with-me' in url:
        path_parts = parsed_url.path.split('/')
        if len(path_parts) > 3:
            return path_parts[-1]
    
    # Direct folder ID
    elif re.match(r'^[A-Za-z0-9_-]{25,}$', url.strip()):
        return url.strip()
    
    logger.error(f"Could not extract folder ID from URL: {url}")
    return None

def get_drive_service():
    """
    Create and return a Google Drive service object
    
    Returns:
        googleapiclient.discovery.Resource: Google Drive service object
    """
    # Check if credentials file exists
    creds_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'credentials.json')
    
    if not os.path.exists(creds_file):
        logger.error("Google Drive credentials file not found")
        raise FileNotFoundError("Google Drive credentials file not found. Please add credentials.json to the application directory.")
    
    # Create credentials from service account file
    credentials = service_account.Credentials.from_service_account_file(
        creds_file, scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    
    # Build the Drive service
    service = build('drive', 'v3', credentials=credentials)
    return service

def list_files_in_folder(folder_id):
    """
    List all files in a Google Drive folder
    
    Args:
        folder_id: Google Drive folder ID
        
    Returns:
        list: List of file metadata dictionaries
    """
    try:
        service = get_drive_service()
        
        # Query for files in the folder, handling pagination
        all_files = []
        page_token = None
        query = f"'{folder_id}' in parents and trashed = false"
        
        while True:
            results = service.files().list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType)",
                pageSize=100, # Fetch 100 at a time
                pageToken=page_token
            ).execute()
            
            files_on_page = results.get('files', [])
            all_files.extend(files_on_page)
            logger.info(f"Fetched {len(files_on_page)} files (total: {len(all_files)}) from Google Drive folder {folder_id}")
            
            page_token = results.get('nextPageToken', None)
            if page_token is None:
                break # No more pages

        files = all_files # Use the complete list
        logger.info(f"Found a total of {len(files)} files in Google Drive folder {folder_id}")

        # Filter for document types we can process
        supported_mimetypes = [
            'text/plain',
            'application/pdf',
            'application/vnd.google-apps.document',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'application/vnd.google-apps.spreadsheet',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel',
            'application/vnd.google-apps.presentation',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'application/vnd.ms-powerpoint',
            'text/csv',
            'text/markdown',
            'application/json',
            'application/xml',
            'text/html',
            'text/javascript',
            'application/javascript'
        ]
        
        # Log all files for debugging
        for f in files:
            logger.info(f"Found file: {f.get('name')} with MIME type: {f.get('mimeType')}")
        
        # Accept all files for now to debug the issue
        document_files = files
        
        # Log the files we're going to process
        logger.info(f"Will process the following files:")
        for f in document_files:
            logger.info(f"- {f.get('name')} ({f.get('mimeType')})")
        
        logger.info(f"Found {len(document_files)} document files in Google Drive folder {folder_id}")
        return document_files
        
    except Exception as e:
        logger.error(f"Error listing files in Google Drive folder: {str(e)}")
        raise

def download_file(file_id, file_name):
    """
    Download a file from Google Drive
    
    Args:
        file_id: Google Drive file ID
        file_name: Name to save the file as
        
    Returns:
        bytes: File content as bytes
    """
    try:
        service = get_drive_service()
        
        # Get file metadata to determine export method
        file_metadata = service.files().get(fileId=file_id, fields='mimeType').execute()
        mime_type = file_metadata.get('mimeType', '')
        
        # For Google Docs, we need to export them
        if mime_type == 'application/vnd.google-apps.document':
            response = service.files().export(
                fileId=file_id,
                mimeType='application/pdf'
            ).execute()
            content = response
            
        # For regular files, we can download directly
        else:
            request = service.files().get_media(fileId=file_id)
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
                logger.info(f"Download {int(status.progress() * 100)}% complete for {file_name}")
            
            content = file_content.getvalue()
        
        logger.info(f"Successfully downloaded {file_name} from Google Drive")
        return content
        
    except Exception as e:
        logger.error(f"Error downloading file from Google Drive: {str(e)}")
        raise

def process_drive_folder(folder_url, max_depth=3):
    """
    Process all files in a Google Drive folder, including nested folders
    
    Args:
        folder_url: Google Drive folder URL
        max_depth: Maximum depth for recursive folder processing
        
    Returns:
        list: List of dictionaries with file name and content
    """
    # Extract folder ID from URL
    folder_id = extract_folder_id_from_url(folder_url)
    if not folder_id:
        raise ValueError(f"Invalid Google Drive folder URL: {folder_url}")
    
    # Process folder recursively
    return process_folder_recursive(folder_id, "", max_depth)

def process_folder_recursive(folder_id, path_prefix="", depth=0, max_depth=3):
    """
    Process a folder recursively
    
    Args:
        folder_id: Google Drive folder ID
        path_prefix: Prefix for file names to indicate folder structure
        depth: Current recursion depth
        max_depth: Maximum recursion depth
        
    Returns:
        list: List of dictionaries with file name and content
    """
    if depth > max_depth:
        logger.warning(f"Maximum recursion depth reached for folder {folder_id}")
        return []
    
    # List files in the folder
    files = list_files_in_folder(folder_id)
    if not files:
        logger.warning(f"No files found in Google Drive folder: {folder_id}")
        return []
    
    # Download each file
    downloaded_files = []
    for file in files:
        try:
            file_id = file.get('id')
            file_name = file.get('name')
            mime_type = file.get('mimeType')
            
            # If it's a folder, process it recursively
            if mime_type == 'application/vnd.google-apps.folder':
                logger.info(f"Processing nested folder: {file_name} ({file_id})")
                nested_path = path_prefix + file_name + "/"
                nested_files = process_folder_recursive(file_id, nested_path, depth + 1, max_depth)
                downloaded_files.extend(nested_files)
                continue
            
            # For regular files, download content
            try:
                logger.info(f"Downloading file: {file_name} ({mime_type})")
                content = download_file(file_id, file_name)
                
                # Use path_prefix to create a hierarchical file name
                prefixed_name = path_prefix + file_name
                
                downloaded_files.append({
                    'name': prefixed_name,
                    'content': content
                })
                logger.info(f"Successfully downloaded: {prefixed_name}")
                
            except Exception as download_e:
                logger.error(f"Error downloading file {file_name}: {str(download_e)}")
                # Continue with other files
            
        except Exception as e:
            logger.error(f"Error processing file {file.get('name')}: {str(e)}")
            # Continue with other files
    
    logger.info(f"Successfully processed {len(downloaded_files)} files from Google Drive folder {folder_id}")
    return downloaded_files
