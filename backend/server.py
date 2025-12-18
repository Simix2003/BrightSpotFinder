from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time
from typing import Dict, Any

from models import (
    StartRequest, StatusResponse, RunStatus, RunStats,
    CombinedStatsRequest, CombinedStatsResponse, BatchStatus
)
from data_manager import DataManager
from batch_manager import BatchManager

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app

# Initialize managers
data_manager = DataManager()
batch_manager = BatchManager(data_manager)

# Store active processing threads
processing_threads: Dict[str, threading.Thread] = {}


def process_run(run_id: str):
    """Background thread to process a run"""
    try:
        run_data = batch_manager.active_runs.get(run_id)
        if not run_data:
            return
        
        total_batches = run_data['stats'].batches_total
        last_batch = run_data['last_batch_completed']
        
        # Process remaining batches
        for batch_id in range(last_batch + 1, total_batches + 1):
            try:
                batch_manager.process_batch(run_id, batch_id)
            except Exception as e:
                print(f"Error in batch {batch_id}: {e}")
                # Continue with next batch
        
        # Mark run as completed
        batch_manager.complete_run(run_id)
        
    except Exception as e:
        print(f"Error processing run {run_id}: {e}")
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
        start_request = StartRequest(**data)
        
        # Validate paths
        import os
        if not os.path.exists(start_request.model_path):
            return jsonify({'error': 'Model file not found'}), 400
        if not os.path.exists(start_request.input_dir):
            return jsonify({'error': 'Input directory not found'}), 400
        
        # Generate run ID
        run_id = data_manager.generate_run_id()
        
        # Create run
        try:
            batch_manager.create_run(
                run_id=run_id,
                model_path=start_request.model_path,
                input_dir=start_request.input_dir,
                output_dir=start_request.output_dir,
                confidence=start_request.confidence,
                run_name=start_request.run_name,
                batch_size=start_request.batch_size
            )
        except Exception as e:
            return jsonify({'error': str(e)}), 400
        
        # Start processing in background thread
        thread = threading.Thread(target=process_run, args=(run_id,))
        thread.daemon = True
        thread.start()
        processing_threads[run_id] = thread
        
        return jsonify({
            'run_id': run_id,
            'message': 'Processing started'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
    print("Starting BrightSpot Detector Server...")
    print("Server will be available at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

