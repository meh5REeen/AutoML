from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import pickle
import json
import pandas as pd
from app.session_manager import get_session_path

router = APIRouter()

class PredictionInput(BaseModel):
    """Input data for making predictions"""
    features: dict  # Dictionary of feature_name: value

@router.post("/single")
def predict_single(session_id: str, input_data: PredictionInput):
    """
    Make a prediction on a single input using the best trained model
    
    Query Parameters:
    - session_id: Session ID
    
    Request Body:
    - features: Dictionary of feature values
    
    Returns:
        Prediction result with confidence/probability
    """
    session_path = get_session_path(session_id)
    
    # Check if session exists
    if not os.path.exists(session_path):
        raise HTTPException(404, f"Session {session_id} not found")
    
    # Load best model
    model_path = os.path.join(session_path, "best_model.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(404, "No trained model found. Please train a model first.")
    
    try:
        with open(model_path, 'rb') as f:
            best_model = pickle.load(f)
    except Exception as e:
        raise HTTPException(500, f"Error loading model: {str(e)}")
    
    # Load model metadata
    metadata_path = os.path.join(session_path, "model_metadata.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(404, "Model metadata not found")
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        raise HTTPException(500, f"Error loading metadata: {str(e)}")
    
    # Load preprocessing transformers
    transformers_path = os.path.join(session_path, "transformers.pkl")
    transformers = None
    if os.path.exists(transformers_path):
        try:
            with open(transformers_path, 'rb') as f:
                transformers = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load transformers: {str(e)}")
    
    # Prepare input data
    try:
        # Create DataFrame from input
        input_df = pd.DataFrame([input_data.features])
        
        # Ensure columns are in the same order as training
        feature_names = metadata.get('feature_names', [])
        if feature_names:
            # Reorder columns to match training data
            input_df = input_df[feature_names]
        
        # Apply transformations if available
        if transformers:
            # Apply encoding
            if 'encoder' in transformers and transformers['encoder']:
                try:
                    input_df = transformers['encoder'].transform(input_df)
                    if hasattr(input_df, 'toarray'):  # If sparse matrix
                        input_df = pd.DataFrame(input_df.toarray())
                except Exception as e:
                    print(f"Warning: Encoding failed: {str(e)}")
            
            # Apply scaling
            if 'scaler' in transformers and transformers['scaler']:
                try:
                    input_df = pd.DataFrame(
                        transformers['scaler'].transform(input_df),
                        columns=input_df.columns
                    )
                except Exception as e:
                    print(f"Warning: Scaling failed: {str(e)}")
        
        # Make prediction
        prediction = best_model.predict(input_df)[0]
        
        # Get probability/confidence if available
        confidence = None
        probabilities = None
        if hasattr(best_model, 'predict_proba'):
            try:
                proba = best_model.predict_proba(input_df)[0]
                probabilities = proba.tolist()
                confidence = float(max(proba))
            except:
                pass
        
        # Convert prediction to native Python type
        if hasattr(prediction, 'item'):
            prediction = prediction.item()
        else:
            prediction = str(prediction)
        
        # Get class labels if available
        class_labels = metadata.get('class_labels', [])
        predicted_label = None
        if class_labels and isinstance(prediction, (int, float)):
            try:
                predicted_label = class_labels[int(prediction)]
            except (IndexError, ValueError):
                predicted_label = str(prediction)
        
        return {
            "prediction": prediction,
            "predicted_label": predicted_label or str(prediction),
            "confidence": confidence,
            "probabilities": probabilities,
            "model_name": metadata.get('model_name', 'Unknown'),
            "class_labels": class_labels
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error making prediction: {str(e)}")


@router.get("/features")
def get_feature_info(session_id: str):
    """
    Get comprehensive information about features needed for prediction
    
    Query Parameters:
    - session_id: Session ID
    
    Returns:
        Detailed feature information including types, examples, ranges, and categorical mappings
    """
    session_path = get_session_path(session_id)
    
    # Check if session exists
    if not os.path.exists(session_path):
        raise HTTPException(404, f"Session {session_id} not found")
    
    # Load model metadata
    metadata_path = os.path.join(session_path, "model_metadata.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(404, "Model metadata not found. Please train a model first.")
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        feature_names = metadata.get('feature_names', [])
        
        # Load original dataset to get data types and sample values
        csv_path = os.path.join(session_path, "dataset.csv")
        excel_path = os.path.join(session_path, "dataset.xlsx")
        
        df_original = None
        if os.path.exists(csv_path):
            try:
                df_original = pd.read_csv(csv_path, encoding="utf-8")
            except UnicodeDecodeError:
                from charset_normalizer import from_path
                detected = from_path(csv_path).best()
                encoding = detected.encoding if detected else "latin-1"
                df_original = pd.read_csv(csv_path, encoding=encoding)
        elif os.path.exists(excel_path):
            df_original = pd.read_excel(excel_path)
        
        # Load categorical mappings if available
        categorical_mappings = {}
        mappings_path = os.path.join(session_path, "categorical_mappings.json")
        if os.path.exists(mappings_path):
            with open(mappings_path, 'r') as f:
                categorical_mappings = json.load(f)
        
        # Build detailed feature information
        feature_info = {}
        
        for feature_name in feature_names:
            info = {
                "name": feature_name,
                "original_dtype": "unknown",
                "data_category": "numeric",
                "sample_values": [],
                "description": ""
            }
            
            # Get information from original dataset if available
            if df_original is not None and feature_name in df_original.columns:
                col_data = df_original[feature_name]
                dtype_name = str(col_data.dtype)
                info["original_dtype"] = dtype_name
                
                # Get sample values (non-null, unique)
                sample_vals = col_data.dropna().unique()[:5].tolist()
                info["sample_values"] = [str(v) for v in sample_vals]
                
                # Determine data category and build description
                if dtype_name in ['object', 'category']:
                    # Categorical text
                    info["data_category"] = "categorical_text"
                    unique_vals = col_data.dropna().unique().tolist()
                    info["unique_values"] = [str(v) for v in unique_vals[:10]]  # Limit to 10
                    
                    # Check if this was encoded
                    if feature_name in categorical_mappings:
                        mapping_info = categorical_mappings[feature_name]
                        if mapping_info.get("type") == "ordinal":
                            info["data_category"] = "categorical_encoded"
                            info["categorical_mapping"] = mapping_info.get("mapping", {})
                            # Build description
                            mapping_str = ", ".join([f"{k}={v}" for k, v in list(info["categorical_mapping"].items())[:5]])
                            info["description"] = f"Category (encoded): {mapping_str}"
                        elif mapping_info.get("type") == "onehot":
                            info["description"] = f"Category: {', '.join([str(v) for v in unique_vals[:5]])}"
                    else:
                        info["description"] = f"Category: {', '.join([str(v) for v in unique_vals[:5]])}"
                
                elif dtype_name in ['int64', 'float64', 'int32', 'float32']:
                    # Numeric
                    info["data_category"] = "numeric"
                    min_val = float(col_data.min())
                    max_val = float(col_data.max())
                    info["value_range"] = {"min": min_val, "max": max_val}
                    
                    # Check if it's actually a categorical encoded as numeric
                    unique_count = col_data.nunique()
                    if unique_count <= 10:  # Likely categorical
                        info["data_category"] = "categorical_encoded"
                        unique_vals = sorted(col_data.dropna().unique().tolist())
                        info["unique_values"] = [str(int(v)) if v == int(v) else str(v) for v in unique_vals]
                        info["description"] = f"Category (numeric): Enter one of {', '.join(info['unique_values'])}"
                    else:
                        info["description"] = f"Numeric: Enter a number between {min_val:.2f} and {max_val:.2f}"
            
            feature_info[feature_name] = info
        
        return {
            "feature_names": feature_names,
            "feature_info": feature_info,
            "class_labels": metadata.get('class_labels', []),
            "model_name": metadata.get('model_name', 'Unknown')
        }
    except Exception as e:
        raise HTTPException(500, f"Error loading feature info: {str(e)}")



@router.get("/download-model")
def download_model(session_id: str):
    """
    Download the trained model pickle file
    
    Query Parameters:
    - session_id: Session ID
    
    Returns:
        Model pickle file for download
    """
    from fastapi.responses import FileResponse
    
    session_path = get_session_path(session_id)
    
    # Check if session exists
    if not os.path.exists(session_path):
        raise HTTPException(404, f"Session {session_id} not found")
    
    # Check if model exists
    model_path = os.path.join(session_path, "best_model.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(404, "No trained model found. Please train a model first.")
    
    # Return the model file for download
    return FileResponse(
        path=model_path,
        media_type="application/octet-stream",
        filename=f"model_{session_id}.pkl"
    )
