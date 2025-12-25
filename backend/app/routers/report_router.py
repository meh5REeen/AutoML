from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import os
from app.session_manager import get_session_path
from app.services.report import ReportGenerator
from datetime import datetime
import io
import tempfile

router = APIRouter()


@router.get("/generate")
def generate_report(
    session_id: str,
    format: str = Query("markdown", description="Report format: markdown, html, or pdf")
):
    """
    Generate a comprehensive AutoML report for a session
    
    Query Parameters:
    - session_id: Session ID to generate report for
    - format: Output format (markdown, html, pdf) - default: markdown
    
    Returns:
        Report file or content in requested format
    """
    # Validate session exists
    session_path = get_session_path(session_id)
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Validate format
    valid_formats = ["markdown", "html", "pdf"]
    if format.lower() not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format. Must be one of: {', '.join(valid_formats)}"
        )
    
    try:
        # Generate report
        generator = ReportGenerator(session_id)
        
        if format.lower() == "pdf":
            # Generate PDF as bytes
            pdf_content = generator.generate_report(format=format.lower())
            
            # Save to temp file
            temp_dir = os.path.join(session_path, "reports")
            os.makedirs(temp_dir, exist_ok=True)
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            full_path = os.path.join(temp_dir, filename)
            
            with open(full_path, 'wb') as f:
                f.write(pdf_content)
            
            return FileResponse(
                path=full_path,
                filename=filename,
                media_type="application/pdf"
            )
        else:
            # Generate HTML or Markdown as text
            report_content = generator.generate_report(format=format.lower())
            
            if format.lower() == "markdown":
                filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                media_type = "text/markdown"
            elif format.lower() == "html":
                filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                media_type = "text/html"
            
            # Save report to session folder
            report_path = os.path.join(session_path, "reports")
            os.makedirs(report_path, exist_ok=True)
            full_report_path = os.path.join(report_path, filename)
            
            with open(full_report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            return FileResponse(
                path=full_report_path,
                filename=filename,
                media_type=media_type
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


@router.get("/preview")
def preview_report(
    session_id: str,
    format: str = Query("markdown", description="Report format: markdown or html")
):
    """
    Preview a report without downloading as file
    
    Query Parameters:
    - session_id: Session ID to preview report for
    - format: Output format (markdown or html) - default: markdown
    
    Returns:
        Report content as JSON with preview
    """
    # Validate session exists
    session_path = get_session_path(session_id)
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Validate format
    valid_formats = ["markdown", "html"]
    if format.lower() not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format. Must be one of: {', '.join(valid_formats)}"
        )
    
    try:
        # Generate report
        generator = ReportGenerator(session_id)
        report_content = generator.generate_report(format=format.lower())
        
        return {
            "status": "success",
            "session_id": session_id,
            "format": format.lower(),
            "generated_at": datetime.now().isoformat(),
            "content": report_content
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


@router.get("/status")
def report_status(session_id: str):
    """
    Check what data is available for report generation
    
    Query Parameters:
    - session_id: Session ID to check
    
    Returns:
        Status of available report data
    """
    session_path = get_session_path(session_id)
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    status = {
        "session_id": session_id,
        "data_available": {}
    }
    
    # Check what files exist
    if os.path.exists(os.path.join(session_path, "dataset.csv")):
        status["data_available"]["original_dataset"] = True
    
    if os.path.exists(os.path.join(session_path, "data_cleaned.csv")):
        status["data_available"]["cleaned_dataset"] = True
    
    if os.path.exists(os.path.join(session_path, "preprocessing_metadata.json")):
        status["data_available"]["preprocessing_metadata"] = True
    
    if os.path.exists(os.path.join(session_path, "model_results.json")):
        status["data_available"]["model_results"] = True
    
    # Check if report can be generated
    status["can_generate_report"] = (
        status["data_available"].get("original_dataset", False) or
        status["data_available"].get("cleaned_dataset", False)
    )
    
    return status
