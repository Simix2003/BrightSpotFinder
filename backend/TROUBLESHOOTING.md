# Troubleshooting Guide: YOLO Inference Not Running

## Problem
The backend .exe is running (health checks work), but YOLO inference doesn't seem to be executing.

## Common Causes

### 1. Model File Not Found
**Symptoms:** Error in logs: "Model file not found"
**Solution:** 
- Ensure the model path is an **absolute path** when running as .exe
- Relative paths may not work correctly when running as executable
- Verify the model file exists at the specified location

### 2. YOLO Model Loading Fails
**Symptoms:** Error in logs: "Failed to load YOLO model"
**Possible causes:**
- Missing CUDA/GPU drivers (YOLO will fall back to CPU, but may fail if dependencies are missing)
- Corrupted model file
- Missing PyTorch dependencies in the exe

**Solution:**
- Check the log file: `logs/inference_engine.log`
- Ensure the model file is not corrupted
- Try running the model on the development machine first to verify it works

### 3. Silent Thread Failures
**Symptoms:** Processing starts but never completes, no errors visible
**Solution:**
- Check the log files in the `logs/` directory:
  - `logs/server.log` - Server and API endpoint logs
  - `logs/inference_engine.log` - Model loading and inference logs
- Look for exceptions or errors that might have been silently caught

### 4. Path Issues in .exe
**Symptoms:** Files not found errors
**Solution:**
- Use absolute paths for all file operations when running as .exe
- The working directory may differ when running as executable
- Consider bundling the model file with the exe (see build_exe.py comments)

### 5. CUDA/GPU Issues
**Symptoms:** Model loads but inference fails or is very slow
**Solution:**
- YOLO will automatically use CPU if CUDA is not available
- Check if CUDA is available: The logs will show if GPU is being used
- For CPU-only systems, inference will work but be slower

## Debugging Steps

1. **Check Log Files**
   - Navigate to the `logs/` directory (created next to the exe)
   - Check `server.log` for API calls and errors
   - Check `inference_engine.log` for model loading and inference errors

2. **Verify Model Path**
   - When calling `/api/start`, ensure `model_path` is an absolute path
   - Example: `"C:\\Models\\yolov8n_residual\\weights\\best.pt"` (Windows)
   - Not: `"Models/yolov8n_residual/weights/best.pt"` (relative)

3. **Test Model Loading**
   - Try loading the model manually on the target PC using Python:
     ```python
     from ultralytics import YOLO
     model = YOLO("path/to/model.pt")
     ```
   - This will reveal if there are dependency issues

4. **Check API Response**
   - When calling `/api/start`, check the response:
     - If it returns an error, check the error message
     - If it returns `run_id`, check the status with `/api/status/<run_id>`

5. **Monitor Background Thread**
   - The processing happens in a background thread
   - Check the logs for thread start messages
   - Look for "Starting processing for run" messages

## Log File Locations

When running as .exe, log files are created in:
- `logs/server.log` - Server operations
- `logs/inference_engine.log` - Inference operations

The logs directory is created automatically next to where the exe is running.

## What to Look For in Logs

1. **Model Loading:**
   ```
   Initializing InferenceEngine with model path: ...
   Model file found. Attempting to load YOLO model...
   YOLO model loaded successfully
   ```

2. **Processing Start:**
   ```
   Starting processing for run ...
   Run ...: Processing batch 1/...
   ```

3. **Errors:**
   - Look for lines starting with `ERROR`
   - Check stack traces (lines with `Traceback`)
   - Note any `FileNotFoundError` or `RuntimeError` messages

## Next Steps

After adding logging, rebuild the .exe:
```bash
cd backend
python build_exe.py
```

Then run the new .exe and check the log files for detailed error information.
