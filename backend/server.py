from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time
import logging
import os
from typing import Dict, Any

from models import (
    StartRequest, StatusResponse, RunStatus, RunStats,
    CombinedStatsRequest, CombinedStatsResponse, BatchStatus
)
from data_manager import DataManager
from batch_manager import BatchManager

# Set up logging
# Handle both script execution and PyInstaller exe execution
import sys
if getattr(sys, 'frozen', False):
    # Running as compiled exe
    base_dir = os.path.dirname(sys.executable)
else:
    # Running as script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(base_dir)  # Go up one level from backend/

log_dir = os.path.join(base_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'server.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app

# Initialize managers
data_manager = DataManager()
batch_manager = BatchManager(data_manager)

# Store active processing threads
processing_threads: Dict[str, threading.Thread] = {}


def process_run(run_id: str):
    """Background thread to process a run"""
    logger.info(f"Starting processing for run {run_id}")
    try:
        run_data = batch_manager.active_runs.get(run_id)
        if not run_data:
            logger.warning(f"Run {run_id} not found in active_runs")
            return
        
        total_batches = run_data['stats'].batches_total
        last_batch = run_data['last_batch_completed']
        logger.info(f"Run {run_id}: Processing {total_batches - last_batch} batches (starting from batch {last_batch + 1})")
        
        # Process remaining batches
        for batch_id in range(last_batch + 1, total_batches + 1):
            try:
                logger.info(f"Run {run_id}: Processing batch {batch_id}/{total_batches}")
                batch_manager.process_batch(run_id, batch_id)
                logger.info(f"Run {run_id}: Batch {batch_id} completed successfully")
            except Exception as e:
                logger.error(f"Run {run_id}: Error in batch {batch_id}: {e}", exc_info=True)
                # Continue with next batch
        
        # Mark run as completed
        logger.info(f"Run {run_id}: All batches completed, marking as completed")
        batch_manager.complete_run(run_id)
        logger.info(f"Run {run_id}: Successfully completed")
        
    except Exception as e:
        logger.error(f"Run {run_id}: Error processing run: {e}", exc_info=True)
        # Mark as failed
        with batch_manager.lock:
            if run_id in batch_manager.active_runs:
                batch_manager.active_runs[run_id]['status'] = RunStatus.FAILED


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Server is running'}), 200


@app.route('/api/start', methods=['POST'])
def start_processing():
    """Start a new processing run"""
    try:
        data = request.json
        logger.info(f"Received start request: {data}")
        start_request = StartRequest(**data)
        
        # Validate paths
        import os
        logger.info(f"Validating model path: {start_request.model_path}")
        if not os.path.exists(start_request.model_path):
            error_msg = f'Model file not found: {start_request.model_path}'
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 400
        
        logger.info(f"Validating input directory: {start_request.input_dir}")
        if not os.path.exists(start_request.input_dir):
            error_msg = f'Input directory not found: {start_request.input_dir}'
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 400
        
        # Generate run ID
        run_id = data_manager.generate_run_id()
        logger.info(f"Generated run_id: {run_id}")
        
        # Create run
        try:
            logger.info(f"Creating run {run_id} with model: {start_request.model_path}")
            batch_manager.create_run(
                run_id=run_id,
                model_path=start_request.model_path,
                input_dir=start_request.input_dir,
                output_dir=start_request.output_dir,
                confidence=start_request.confidence,
                run_name=start_request.run_name,
                batch_size=start_request.batch_size
            )
            logger.info(f"Run {run_id} created successfully")
        except Exception as e:
            error_msg = f"Failed to create run: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return jsonify({'error': error_msg}), 400
        
        # Start processing in background thread
        logger.info(f"Starting background thread for run {run_id}")
        thread = threading.Thread(target=process_run, args=(run_id,))
        thread.daemon = True
        thread.start()
        processing_threads[run_id] = thread
        logger.info(f"Background thread started for run {run_id}")
        
        return jsonify({
            'run_id': run_id,
            'message': 'Processing started'
        }), 200
        
    except Exception as e:
        error_msg = f"Unexpected error in start_processing: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({'error': error_msg}), 500


@app.route('/api/status/<run_id>', methods=['GET'])
def get_status(run_id: str):
    """Get status of a run"""
    try:
        status_data = batch_manager.get_run_status(run_id)
        if not status_data:
            return jsonify({'error': 'Run not found'}), 404
        
        stats = status_data['stats']
        
        # Convert status enum to string if needed
        status_value = status_data['status']
        if hasattr(status_value, 'value'):
            status_value = status_value.value
        
        response = StatusResponse(
            run_id=run_id,
            status=status_value,
            progress=status_data.get('progress', 0.0),
            current_batch=status_data.get('current_batch', 0),
            total_batches=status_data.get('total_batches', 0),
            stats=stats
        )
        
        return jsonify(response.model_dump(mode='json', exclude_none=True)), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats/<run_id>', methods=['GET'])
def get_stats(run_id: str):
    """Get statistics for a completed run"""
    try:
        run_data = data_manager.load_run(run_id)
        if not run_data:
            return jsonify({'error': 'Run not found'}), 404
        
        return jsonify({
            'run_id': run_id,
            'run_name': run_data.run_name,
            'timestamp': run_data.timestamp.isoformat(),
            'stats': run_data.stats.model_dump(mode='json', exclude_none=True)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get history of all runs"""
    try:
        limit = request.args.get('limit', type=int)
        runs = data_manager.get_all_runs(limit=limit)
        
        history = []
        for run in runs:
            history.append({
                'run_id': run.run_id,
                'run_name': run.run_name,
                'timestamp': run.timestamp.isoformat(),
                'status': run.status.value,
                'stats': run.stats.model_dump(mode='json', exclude_none=True)
            })
        
        return jsonify({'runs': history}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats/combined', methods=['POST'])
def get_combined_stats():
    """Get combined statistics for multiple runs"""
    try:
        data = request.json
        request_obj = CombinedStatsRequest(**data)
        
        runs_data = []
        total_images = 0
        total_bright_spots = 0
        total_without_spot = 0
        success_rates = []
        bright_spot_rates = []
        
        for run_id in request_obj.run_ids:
            run_data = data_manager.load_run(run_id)
            if run_data:
                stats = run_data.stats
                runs_data.append({
                    'run_id': run_id,
                    'run_name': run_data.run_name,
                    'timestamp': run_data.timestamp.isoformat(),
                    'stats': stats.model_dump(mode='json', exclude_none=True)
                })
                
                total_images += stats.total_images
                total_bright_spots += stats.images_with_bright_spot
                total_without_spot += stats.images_without_bright_spot
                if stats.total_images > 0:
                    success_rates.append(stats.success_rate)
                    bright_spot_rates.append(stats.bright_spot_rate)
        
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.0
        avg_bright_spot_rate = sum(bright_spot_rates) / len(bright_spot_rates) if bright_spot_rates else 0.0
        
        response = CombinedStatsResponse(
            total_runs=len(runs_data),
            total_images_processed=total_images,
            total_bright_spots_found=total_bright_spots,
            total_images_without_bright_spot=total_without_spot,
            average_success_rate=avg_success_rate,
            average_bright_spot_rate=avg_bright_spot_rate,
            runs=runs_data
        )
        
        return jsonify(response.model_dump(mode='json', exclude_none=True)), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch/status/<run_id>', methods=['GET'])
def get_batch_status(run_id: str):
    """Get detailed batch status for a run"""
    try:
        status_data = batch_manager.get_run_status(run_id)
        if not status_data:
            return jsonify({'error': 'Run not found'}), 404
        
        # Get batch details from active run
        with batch_manager.lock:
            if run_id in batch_manager.active_runs:
                run_data = batch_manager.active_runs[run_id]
                batches = [
                    {
                        'batch_id': b['batch_id'],
                        'status': b['status'],
                        'images_processed': b['images_processed'],
                        'total_images': b['total_images']
                    }
                    for b in run_data['batches']
                ]
            else:
                # Load from disk
                run_record = data_manager.load_run(run_id)
                if run_record:
                    batches = [b.model_dump(mode='json', exclude_none=True) for b in run_record.batches]
                else:
                    batches = []
        
        return jsonify({
            'run_id': run_id,
            'batches': batches,
            'overall_status': status_data['status'].value if hasattr(status_data['status'], 'value') else str(status_data['status'])
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("="*60)
    logger.info("Starting BrightSpot Detector Server...")
    logger.info("Server will be available at http://localhost:5000")
    logger.info(f"Logs will be written to: {log_file}")
    logger.info("="*60)
    print("Starting BrightSpot Detector Server...")
    print("Server will be available at http://localhost:5000")
    print(f"Logs will be written to: {log_file}")
    app.run(host='0.0.0.0', port=5000, debug=False)

