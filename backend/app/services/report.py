import os
import json
import pandas as pd
from datetime import datetime
from app.session_manager import get_session_path
from typing import Optional, Dict, Any
from io import StringIO
import re


class ReportGenerator:
    """Generate comprehensive AutoML reports in multiple formats"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session_path = get_session_path(session_id)
        self.report_data = {}
        self._load_all_metadata()
    
    def _load_all_metadata(self):
        """Load all available metadata from session"""
        # Load preprocessing metadata
        preprocessing_path = os.path.join(self.session_path, "preprocessing_metadata.json")
        if os.path.exists(preprocessing_path):
            with open(preprocessing_path, 'r') as f:
                self.report_data['preprocessing'] = json.load(f)
        
        # Load model results
        model_results_path = os.path.join(self.session_path, "model_results.json")
        if os.path.exists(model_results_path):
            with open(model_results_path, 'r') as f:
                self.report_data['models'] = json.load(f)
        
        # Load dataset
        csv_path = os.path.join(self.session_path, "dataset.csv")
        cleaned_csv_path = os.path.join(self.session_path, "data_cleaned.csv")
        
        if os.path.exists(csv_path):
            self.report_data['original_df'] = pd.read_csv(csv_path)
        
        if os.path.exists(cleaned_csv_path):
            self.report_data['cleaned_df'] = pd.read_csv(cleaned_csv_path)
    
    def generate_report(self, format: str = "pdf") -> str:
        """Generate report in specified format"""
        if format.lower() == "markdown":
            return self._generate_markdown()
        elif format.lower() == "html":
            return self._generate_html()
        elif format.lower() == "pdf":
            return self._generate_pdf()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_markdown(self) -> str:
        """Generate Markdown report"""
        md = []
        md.append("# AutoML Final Report\n")
        md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 1. Dataset Overview
        md.extend(self._section_dataset_overview())
        
        # 2. EDA Findings (if available)
        md.extend(self._section_eda_findings())
        
        # 3. Detected Issues
        md.extend(self._section_detected_issues())
        
        # 4. Preprocessing Decisions
        md.extend(self._section_preprocessing_decisions())
        
        # 5. Model Configurations
        md.extend(self._section_model_configs())
        
        # 6. Model Comparison
        md.extend(self._section_model_comparison())
        
        # 7. Best Model Summary
        md.extend(self._section_best_model())
        
        return "\n".join(md)
    
    def _generate_html(self) -> str:
        """Generate HTML report"""
        md_content = self._generate_markdown()
        
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("  <meta charset='UTF-8'>")
        html.append("  <title>AutoML Final Report</title>")
        html.append("  <style>")
        html.extend(self._get_css_styles())
        html.append("  </style>")
        html.append("</head>")
        html.append("<body>")
        html.append(self._markdown_to_html(md_content))
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)
    
    def _generate_pdf(self) -> bytes:
        """Generate PDF report using reportlab"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib import colors
            from io import BytesIO
            
            pdf_buffer = BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1976d2'),
                spaceAfter=30
            )
            story.append(Paragraph("AutoML Final Report", title_style))
            story.append(Spacer(1, 0.3*inch))
            
            md_content = self._generate_markdown()
            story.extend(self._markdown_to_reportlab(md_content, styles))
            
            doc.build(story)
            pdf_buffer.seek(0)
            return pdf_buffer.getvalue()
        except ImportError:
            html = self._generate_html()
            return html.encode('utf-8')
    
    def _markdown_to_reportlab(self, md_content: str, styles):
        """Convert Markdown to ReportLab story elements"""
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        
        story = []
        lines = md_content.split("\n")
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Handle headers
            if line.startswith("# "):
                story.append(Paragraph(line[2:], styles['Heading1']))
                story.append(Spacer(1, 0.2*inch))
                i += 1
            elif line.startswith("## "):
                story.append(Paragraph(line[3:], styles['Heading2']))
                story.append(Spacer(1, 0.15*inch))
                i += 1
            elif line.startswith("### "):
                story.append(Paragraph(line[4:], styles['Heading3']))
                story.append(Spacer(1, 0.1*inch))
                i += 1
            # Handle tables
            elif line.startswith("|"):
                table_lines = []
                while i < len(lines) and lines[i].startswith("|"):
                    if "---|" not in lines[i]:  # Skip separator
                        cells = [cell.strip() for cell in lines[i].split("|")[1:-1]]
                        table_lines.append(cells)
                    i += 1
                
                if table_lines:
                    # Determine column widths based on content
                    num_cols = len(table_lines[0])
                    col_widths = self._calculate_col_widths(table_lines, num_cols)
                    
                    # Wrap cell content in Paragraph objects for proper text wrapping
                    wrapped_table = []
                    for row in table_lines:
                        wrapped_row = [Paragraph(cell, styles['Normal']) for cell in row]
                        wrapped_table.append(wrapped_row)
                    
                    table = Table(wrapped_table, colWidths=col_widths)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                        ('FONTSIZE', (0, 1), (-1, -1), 8),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
                    ]))
                    story.append(table)
                    story.append(Spacer(1, 0.2*inch))
            # Handle bullet lists
            elif line.startswith("- "):
                list_items = []
                while i < len(lines) and lines[i].startswith("- "):
                    list_items.append(lines[i][2:])
                    i += 1
                for item in list_items:
                    story.append(Paragraph(f"• {self._clean_markdown(item)}", styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
            # Handle regular text
            elif line.strip():
                story.append(Paragraph(self._clean_markdown(line), styles['Normal']))
                story.append(Spacer(1, 0.05*inch))
                i += 1
            else:
                story.append(Spacer(1, 0.1*inch))
                i += 1
        
        return story
    
    def _calculate_col_widths(self, table_lines: list, num_cols: int) -> list:
        """Calculate appropriate column widths based on content"""
        from reportlab.lib.units import inch
        
        # Get max content length for each column
        max_lengths = [0] * num_cols
        for row in table_lines:
            for col_idx, cell in enumerate(row):
                max_lengths[col_idx] = max(max_lengths[col_idx], len(cell))
        
        # Allocate widths proportionally, with minimum widths
        total_width = 7.5 * inch  # Total available width
        min_col_width = 0.8 * inch
        
        # Start with proportional allocation
        total_chars = sum(max_lengths)
        if total_chars == 0:
            total_chars = num_cols
        
        col_widths = [max(min_col_width, (length / total_chars) * total_width) for length in max_lengths]
        
        # Adjust if total exceeds available width
        total_allocated = sum(col_widths)
        if total_allocated > total_width:
            scaling_factor = total_width / total_allocated
            col_widths = [w * scaling_factor for w in col_widths]
        
        return col_widths
    
    def _section_dataset_overview(self) -> list:
        """Generate Dataset Overview section"""
        section = []
        section.append("## 1. Dataset Overview\n")
        
        if 'original_df' in self.report_data:
            df = self.report_data['original_df']
            section.append(f"**Original Dataset:**")
            section.append(f"- Rows: {df.shape[0]}")
            section.append(f"- Columns: {df.shape[1]}")
            section.append(f"- Missing Values: {df.isnull().sum().sum()} ({(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%)")
            section.append(f"- Duplicate Rows: {df.duplicated().sum()}")
            
            # Column info
            section.append(f"\n**Column Summary:**")
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            section.append(f"- Numeric Columns ({len(numeric_cols)}): {', '.join(numeric_cols)}")
            section.append(f"- Categorical Columns ({len(categorical_cols)}): {', '.join(categorical_cols)}")
        
        if 'preprocessing' in self.report_data:
            preprocessing = self.report_data['preprocessing']
            after = preprocessing.get('data_quality_after', {})
            section.append(f"\n**After Preprocessing:**")
            section.append(f"- Rows: {after.get('total_rows', 'N/A')}")
            section.append(f"- Columns: {after.get('total_columns', 'N/A')}")
            section.append(f"- Rows Removed: {preprocessing.get('rows_removed', 0)}")
        
        section.append("\n")
        return section
    
    def _section_eda_findings(self) -> list:
        """Generate EDA Findings section"""
        section = []
        section.append("## 2. EDA Findings\n")
        
        if 'original_df' in self.report_data:
            df = self.report_data['original_df']
            
            # Basic statistics
            section.append("**Numeric Features Statistics:**\n")
            section.append("| Feature | Mean | Median | Std | Min | Max |")
            section.append("|---------|------|--------|-----|-----|-----|")
            
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                mean_val = df[col].mean()
                median_val = df[col].median()
                std_val = df[col].std()
                min_val = df[col].min()
                max_val = df[col].max()
                section.append(f"| {col} | {mean_val:.2f} | {median_val:.2f} | {std_val:.2f} | {min_val:.2f} | {max_val:.2f} |")
            
            section.append("\n")
        
        section.append("\n")
        return section
    
    def _section_detected_issues(self) -> list:
        """Generate Detected Issues section"""
        section = []
        section.append("## 3. Data Quality Issues Detected\n")
        
        if 'preprocessing' in self.report_data:
            preprocessing = self.report_data['preprocessing']
            issues_report = preprocessing.get('issues_detected', {})
            
            severity_summary = issues_report.get('severity_summary', {})
            section.append(f"**Issues Summary:**")
            section.append(f"- High Severity: {severity_summary.get('high', 0)}")
            section.append(f"- Medium Severity: {severity_summary.get('medium', 0)}")
            section.append(f"- Low Severity: {severity_summary.get('low', 0)}")
            
            issues = issues_report.get('issues_detected', [])
            if issues:
                section.append(f"\n**Issues Detected:**\n")
                for issue in issues:
                    issue_type = issue.get('type', 'Unknown').upper()
                    column = issue.get('column', 'N/A')
                    severity = issue.get('severity', 'Unknown').upper()
                    section.append(f"- [{severity}] {issue_type} in '{column}': {issue.get('count', 'N/A')} ({issue.get('percent', 0):.2f}%)")
            
            # recommendations = issues_report.get('recommendations', [])
            # if recommendations:
            #     section.append(f"\n**Recommendations:**")
            #     for rec in recommendations:
            #         section.append(f"- {rec}")
        
        section.append("\n")
        return section
    
    def _section_preprocessing_decisions(self) -> list:
        """Generate Preprocessing Decisions section"""
        section = []
        section.append("## 4. Preprocessing Decisions\n")
        
        if 'preprocessing' in self.report_data:
            preprocessing = self.report_data['preprocessing']
            params = preprocessing.get('parameters', {})
            
            section.append("**Methods Applied:**")
            section.append(f"- Missing Values Strategy: {params.get('missing_strategy', 'N/A')}")
            section.append(f"- Outlier Handling: {params.get('outlier_method', 'N/A')}")
            section.append(f"- Scaling Method: {params.get('scaling_method', 'N/A')}")
            section.append(f"- Encoding Method: {params.get('encoding_method', 'N/A')}")
            section.append(f"- Test Size: {params.get('test_size', 'N/A')}")
            
            before = preprocessing.get('data_quality_before', {})
            after = preprocessing.get('data_quality_after', {})
            
            section.append(f"\n**Impact Summary:**")
            section.append(f"- Rows Removed: {preprocessing.get('rows_removed', 0)}")
            section.append(f"- Missing Values Reduced: {before.get('missing_values', 0)} → {after.get('missing_values', 0)}")
            section.append(f"- Features Modified: {after.get('total_columns', 0)} (from {before.get('total_columns', 0)})")
        
        section.append("\n")
        return section
    
    def _section_model_configs(self) -> list:
        """Generate Model Configurations section"""
        section = []
        section.append("## 5. Model Configurations & Hyperparameters\n")
        
        if 'models' in self.report_data:
            models = self.report_data['models']
            hyperparams = models.get('hyperparameters', {})
            
            section.append("**Training Configuration:**")
            section.append(f"- Test Size: {hyperparams.get('test_size', 'N/A')}")
            section.append(f"- Random State: {hyperparams.get('random_state', 'N/A')}")
            section.append(f"- Hyperparameter Tuning: {'Yes' if hyperparams.get('optimize') else 'No'}")
            
            # Separate tuned and non-tuned models
            model_list = models.get('models', [])
            tuned_models = [m for m in model_list if 'best_params' in m.get('metrics', {})]
            non_tuned_models = [m for m in model_list if 'best_params' not in m.get('metrics', {})]
            
            section.append(f"\n**Models Trained: {len(model_list)} total**\n")
            
            if tuned_models:
                section.append(f"**Hyperparameter Tuned Models ({len(tuned_models)}):**")
                for model in tuned_models:
                    model_name = model.get('name', 'Unknown')
                    metrics = model.get('metrics', {})
                    section.append(f"\n- **{model_name}**")
                    best_params = metrics.get('best_params', {})
                    for param, value in best_params.items():
                        section.append(f"  - {param}: {value}")
            
            if non_tuned_models:
                section.append(f"\n**Non-Tuned Models ({len(non_tuned_models)}):**")
                for model in non_tuned_models[:15]:  # Limit display
                    model_name = model.get('name', 'Unknown')
                    section.append(f"- {model_name} (default parameters)")
        
        section.append("\n")
        return section
    
    def _section_model_comparison(self) -> list:
        """Generate Model Comparison section"""
        section = []
        section.append("## 6. Model Performance Comparison\n")
        
        if 'models' in self.report_data:
            models = self.report_data['models']
            model_list = models.get('models', [])
            
            section.append("| Model | Accuracy | Precision | Recall | F1-Score | Training Time |")
            section.append("|-------|----------|-----------|--------|----------|----------------|")
            
            for model in model_list:
                name = model.get('name', 'Unknown')
                metrics = model.get('metrics', {})
                accuracy = metrics.get('accuracy', 'N/A')
                precision = metrics.get('precision', 'N/A')
                recall = metrics.get('recall', 'N/A')
                f1 = metrics.get('f1_score', 'N/A')
                time_taken = metrics.get('training_time', 'N/A')
                
                if isinstance(accuracy, (int, float)):
                    accuracy = f"{accuracy:.4f}"
                if isinstance(precision, (int, float)):
                    precision = f"{precision:.4f}"
                if isinstance(recall, (int, float)):
                    recall = f"{recall:.4f}"
                if isinstance(f1, (int, float)):
                    f1 = f"{f1:.4f}"
                if isinstance(time_taken, (int, float)):
                    time_taken = f"{time_taken:.2f}s"
                
                section.append(f"| {name} | {accuracy} | {precision} | {recall} | {f1} | {time_taken} |")
        
        section.append("\n")
        return section
    
    def _section_best_model(self) -> list:
        """Generate Best Model Summary section"""
        section = []
        section.append("## 7. Best Model Summary & Justification\n")
        
        if 'models' in self.report_data:
            models = self.report_data['models']
            best_model = models.get('best_model', {})
            
            section.append(f"**Selected Model: {best_model.get('name', 'N/A')}**\n")
            section.append(f"**Reason: {best_model.get('reason', 'N/A')}**\n")
            
            best_name = best_model.get('name', '')
            model_list = models.get('models', [])
            for model in model_list:
                if model.get('name') == best_name:
                    metrics = model.get('metrics', {})
                    section.append(f"\n**Performance Metrics:**")
                    section.append(f"- Accuracy: {metrics.get('accuracy', 'N/A')}")
                    section.append(f"- Precision: {metrics.get('precision', 'N/A')}")
                    section.append(f"- Recall: {metrics.get('recall', 'N/A')}")
                    section.append(f"- F1-Score: {metrics.get('f1_score', 'N/A')}")
                    section.append(f"- ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
                    section.append(f"- Training Time: {metrics.get('training_time', 'N/A')}s")
                    
                    if 'best_params' in metrics:
                        section.append(f"\n**Hyperparameters:**")
                        for param, value in metrics.get('best_params', {}).items():
                            section.append(f"- {param}: {value}")
                    break
        
        section.append("\n")
        return section
    
    def _clean_markdown(self, text: str) -> str:
        """Convert markdown formatting to plain text or HTML formatting"""
        # Bold: **text** → <b>text</b>
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        # Italic: *text* or _text_ → <i>text</i>
        text = re.sub(r'\_(.*?)\_', r'<i>\1</i>', text)
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        # Code: `text` → <code>text</code>
        text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
        return text
    
    def _get_css_styles(self) -> list:
        """Get CSS styles for HTML report"""
        styles = [
            "body {",
            "  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;",
            "  line-height: 1.6;",
            "  color: #333;",
            "  max-width: 1200px;",
            "  margin: 0 auto;",
            "  padding: 20px;",
            "  background-color: #f5f5f5;",
            "}",
            "h1 { color: #1976d2; border-bottom: 3px solid #1976d2; padding-bottom: 10px; }",
            "h2 { color: #1976d2; margin-top: 30px; }",
            "h3 { color: #424242; }",
            "table {",
            "  width: 100%;",
            "  border-collapse: collapse;",
            "  margin: 15px 0;",
            "  background-color: white;",
            "}",
            "th, td {",
            "  border: 1px solid #ddd;",
            "  padding: 12px;",
            "  text-align: left;",
            "}",
            "th { background-color: #1976d2; color: white; }",
            "tr:nth-child(even) { background-color: #f9f9f9; }",
            "tr:hover { background-color: #f0f0f0; }",
            "ul { margin: 10px 0; }",
            "li { margin: 5px 0; }",
            "code { background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; }",
            "b { color: #1976d2; }",
        ]
        return styles
    
    def _markdown_to_html(self, md_content: str) -> str:
        """Convert Markdown to basic HTML"""
        lines = md_content.split("\n")
        result = []
        in_table = False
        in_list = False
        
        for line in lines:
            # Handle tables
            if line.startswith("|"):
                if "---|" in line:
                    continue
                if not in_table:
                    result.append("<table>")
                    in_table = True
                cells = [f"<td>{self._clean_markdown(cell.strip())}</td>" for cell in line.split("|")[1:-1]]
                result.append("<tr>" + "".join(cells) + "</tr>")
            # End table if moving to non-table content
            elif in_table and not line.startswith("|"):
                result.append("</table>")
                in_table = False
                # Continue processing this line
                if line.startswith("## "):
                    if in_list:
                        result.append("</ul>")
                        in_list = False
                    result.append(f"<h2>{self._clean_markdown(line[3:])}</h2>")
                elif line.startswith("# "):
                    if in_list:
                        result.append("</ul>")
                        in_list = False
                    result.append(f"<h1>{self._clean_markdown(line[2:])}</h1>")
                elif line.startswith("- "):
                    if not in_list:
                        result.append("<ul>")
                        in_list = True
                    result.append(f"<li>{self._clean_markdown(line[2:])}</li>")
                elif line.strip():
                    if in_list:
                        result.append("</ul>")
                        in_list = False
                    result.append(f"<p>{self._clean_markdown(line)}</p>")
            # Handle headers
            elif line.startswith("## "):
                if in_list:
                    result.append("</ul>")
                    in_list = False
                result.append(f"<h2>{self._clean_markdown(line[3:])}</h2>")
            elif line.startswith("# "):
                if in_list:
                    result.append("</ul>")
                    in_list = False
                result.append(f"<h1>{self._clean_markdown(line[2:])}</h1>")
            elif line.startswith("### "):
                result.append(f"<h3>{self._clean_markdown(line[4:])}</h3>")
            # Handle lists
            elif line.startswith("- "):
                if not in_list:
                    result.append("<ul>")
                    in_list = True
                result.append(f"<li>{self._clean_markdown(line[2:])}</li>")
            # End list when encountering non-list content
            elif in_list and line.strip() and not line.startswith("- "):
                result.append("</ul>")
                in_list = False
                result.append(f"<p>{self._clean_markdown(line)}</p>")
            # Handle regular text
            elif line.strip():
                result.append(f"<p>{self._clean_markdown(line)}</p>")
        
        # Close any open tags
        if in_table:
            result.append("</table>")
        if in_list:
            result.append("</ul>")
        
        return "\n".join(result)