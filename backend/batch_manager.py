import os
import statistics
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from threading import Lock

from models import BatchInfo, BatchStatus, RunStats, RunStatus
from inference_engine import InferenceEngine
from data_manager import DataManager

logger = logging.getLogger(__name__)


class BatchManager:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.active_runs: Dict[str, Dict[str, Any]] = {}
        self.lock = Lock()
    
    def create_run(
        self,
        run_id: str,
        model_path: str,
        input_dir: str,
        output_dir: str,
        confidence: float,
        run_name: str,
        batch_size: int
    ) -> Dict[str, Any]:
        """Create a new run and divide images into batches"""
        logger.info(f"Creating run {run_id} with model: {model_path}")
        # Initialize inference engine
        try:
            logger.info(f"Initializing InferenceEngine for run {run_id}")
            inference_engine = InferenceEngine(model_path)
            logger.info(f"InferenceEngine initialized successfully for run {run_id}")
        except Exception as e:
            error_msg = f"Failed to initialize InferenceEngine: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
        
        # Get list of images
        logger.info(f"Getting image list from: {input_dir}")
        image_files = inference_engine.get_image_list(input_dir)
        total_images = len(image_files)
        logger.info(f"Found {total_images} images in input directory")
        
        if total_images == 0:
            error_msg = "No images found in input directory"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Divide into batches
        batches = []
        total_batches = (total_images + batch_size - 1) // batch_size  # Ceiling division
        
        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, total_images)
            batch_images = image_files[start_idx:end_idx]
            
            batches.append({
                'batch_id': i + 1,
                'images': batch_images,
                'status': BatchStatus.PENDING.value,
                'images_processed': 0,
                'total_images': len(batch_images)
            })
        
        # Check for existing checkpoint
        checkpoint = self.data_manager.load_checkpoint(run_id)
        last_batch_completed = 0
        images_processed = []
        restored_stats = {
            'images_with_bright_spot': 0,
            'images_without_bright_spot': 0,
            'images_processed_count': 0,
            'total_detections': 0,
            'inference_times': []
        }
        
        if checkpoint:
            last_batch_completed = checkpoint.get('last_batch_completed', 0)
            images_processed = checkpoint.get('images_processed', [])
            partial_stats = checkpoint.get('partial_stats', {})
            restored_stats['images_with_bright_spot'] = partial_stats.get('images_with_bright_spot', 0)
            restored_stats['images_without_bright_spot'] = partial_stats.get('images_without_bright_spot', 0)
            restored_stats['images_processed_count'] = partial_stats.get('images_processed_count', len(images_processed))
            restored_stats['total_detections'] = partial_stats.get('total_detections', 0)
            restored_stats['inference_times'] = partial_stats.get('inference_times', [])
            print(f"Found checkpoint for run {run_id}, resuming from batch {last_batch_completed + 1}")
        
        # Create run data structure
        run_data = {
            'run_id': run_id,
            'model_path': model_path,
            'input_dir': input_dir,
            'output_dir': output_dir,
            'confidence': confidence,
            'run_name': run_name,
            'batch_size': batch_size,
            'inference_engine': inference_engine,
            'batches': batches,
            'total_images': total_images,
            'last_batch_completed': last_batch_completed,
            'images_processed': images_processed,
            'inference_times': restored_stats['inference_times'].copy(),  # Restore inference times from checkpoint
            'stats': RunStats(
                total_images=total_images,
                batches_total=total_batches,
                batches_completed=last_batch_completed,
                images_with_bright_spot=restored_stats['images_with_bright_spot'],  # Restore from checkpoint
                images_without_bright_spot=restored_stats['images_without_bright_spot'],  # Restore from checkpoint
                images_processed_count=restored_stats['images_processed_count'],  # Restore from checkpoint
                total_detections=restored_stats['total_detections'],  # Restore from checkpoint
                median_inference_time_seconds=statistics.median(restored_stats['inference_times']) if restored_stats['inference_times'] else None,  # Calculate median from restored times
                start_time=datetime.now()
            ),
            'status': RunStatus.RUNNING if last_batch_completed == 0 else RunStatus.RUNNING,
            'timestamp': datetime.now()
        }
        
        with self.lock:
            self.active_runs[run_id] = run_data
        
        return run_data
    
    def process_batch(self, run_id: str, batch_id: int) -> Dict[str, Any]:
        """Process a single batch of images"""
        # Get run data and batch info (with lock)
        with self.lock:
            if run_id not in self.active_runs:
                raise ValueError(f"Run {run_id} not found")
            run_data = self.active_runs[run_id]
            
            # Find the batch
            batch = None
            for b in run_data['batches']:
                if b['batch_id'] == batch_id:
                    batch = b
                    break
            
            if not batch:
                raise ValueError(f"Batch {batch_id} not found for run {run_id}")
            
            if batch['status'] == BatchStatus.COMPLETED.value:
                # Already processed, skip
                return {'status': 'already_completed', 'batch_id': batch_id}
            
            # Update batch status
            batch['status'] = BatchStatus.PROCESSING.value
            batch_start_time = datetime.now()
            
            # Get references to needed objects
            inference_engine = run_data['inference_engine']
            confidence = run_data['confidence']
            output_dir = run_data['output_dir']
            batch_images = batch['images'].copy()  # Copy to work outside lock
        
        # Process images without holding the lock (allows get_run_status to work)
        batch_results = []
        batch_images_with_spot = 0
        batch_total_detections = 0
        
        try:
            for image_path in batch_images:
                image_name = os.path.basename(image_path)
                
                # Check and update with lock
                with self.lock:
                    if run_id not in self.active_runs:
                        raise ValueError(f"Run {run_id} no longer exists")
                    run_data = self.active_runs[run_id]
                    
                    # Skip if already processed (from checkpoint)
                    if image_name in run_data['images_processed']:
                        continue
                
                try:
                    # Process image (outside lock for performance)
                    logger.debug(f"Processing image: {image_name}")
                    image_result = inference_engine.process_image(
                        image_path,
                        confidence_threshold=confidence,
                        inference_conf=0.3,  # Lower threshold for inference, filter later
                        imgsz=1024
                    )
                    logger.debug(f"Image {image_name} processed: has_bright_spot={image_result.has_bright_spot}")
                    
                    # Save annotated image if bright spot detected
                    has_spot = image_result.has_bright_spot
                    detections_count = len(image_result.detections) if has_spot else 0
                    
                    if has_spot:
                        inference_engine.save_annotated_image(
                            image_path,
                            image_result,
                            output_dir
                        )
                        batch_images_with_spot += 1
                        batch_total_detections += detections_count
                    
                    batch_results.append(image_result)
                    
                    # Update with lock - update statistics immediately after each image
                    with self.lock:
                        if run_id not in self.active_runs:
                            raise ValueError(f"Run {run_id} no longer exists")
                        run_data = self.active_runs[run_id]
                        run_data['images_processed'].append(image_name)
                        batch['images_processed'] += 1
                        # Update total images count for progress calculation
                        run_data['stats'].total_images = run_data['total_images']
                        
                        # Track inference time
                        inference_time = image_result.inference_time_seconds
                        if inference_time is not None:
                            run_data['inference_times'].append(inference_time)
                        
                        # Increment images_processed_count
                        run_data['stats'].images_processed_count += 1
                        
                        # Update statistics immediately after each image
                        if has_spot:
                            run_data['stats'].images_with_bright_spot += 1
                            run_data['stats'].total_detections += detections_count
                        else:
                            # Increment images_without_bright_spot (don't calculate as total - with)
                            run_data['stats'].images_without_bright_spot += 1
                        
                        # Calculate median inference time
                        stats = run_data['stats']
                        if run_data['inference_times']:
                            stats.median_inference_time_seconds = statistics.median(run_data['inference_times'])
                        else:
                            stats.median_inference_time_seconds = None
                        
                        # Calculate and update rates after each image
                        processed_count = stats.images_processed_count
                        if processed_count > 0:
                            stats.success_rate = (stats.images_without_bright_spot / processed_count * 100)
                            stats.bright_spot_rate = (stats.images_with_bright_spot / processed_count * 100)
                        else:
                            stats.success_rate = 0.0
                            stats.bright_spot_rate = 0.0
                    
                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {e}", exc_info=True)
                    # Continue with next image
                    continue
            
            # Update batch status and statistics (with lock)
            with self.lock:
                if run_id not in self.active_runs:
                    raise ValueError(f"Run {run_id} no longer exists")
                run_data = self.active_runs[run_id]
                batch = None
                for b in run_data['batches']:
                    if b['batch_id'] == batch_id:
                        batch = b
                        break
                
                if batch:
                    batch['status'] = BatchStatus.COMPLETED.value
                    batch['end_time'] = datetime.now()
                
                # Update batch completion status (statistics already updated after each image)
                run_data['stats'].batches_completed = batch_id
                run_data['last_batch_completed'] = batch_id
                
                # Ensure rates are up to date (use processed_count, not total_images)
                stats = run_data['stats']
                processed_count = stats.images_processed_count
                if processed_count > 0:
                    stats.success_rate = (stats.images_without_bright_spot / processed_count * 100)
                    stats.bright_spot_rate = (stats.images_with_bright_spot / processed_count * 100)
                else:
                    stats.success_rate = 0.0
                    stats.bright_spot_rate = 0.0
                
                # Save checkpoint
                self._save_checkpoint(run_id, run_data)
            
            return {
                'status': 'completed',
                'batch_id': batch_id,
                'images_processed': batch['images_processed'],
                'images_with_spot': batch_images_with_spot
            }
            
        except Exception as e:
            # Update batch status to failed (with lock)
            with self.lock:
                if run_id in self.active_runs:
                    run_data = self.active_runs[run_id]
                    batch = None
                    for b in run_data['batches']:
                        if b['batch_id'] == batch_id:
                            batch = b
                            break
                    if batch:
                        batch['status'] = BatchStatus.FAILED.value
            logger.error(f"Error processing batch {batch_id} for run {run_id}: {e}", exc_info=True)
            raise
    
    def _save_checkpoint(self, run_id: str, run_data: Dict[str, Any]):
        """Save checkpoint data"""
        checkpoint_data = {
            'run_id': run_id,
            'last_batch_completed': run_data['last_batch_completed'],
            'images_processed': run_data['images_processed'],
            'partial_stats': {
                'images_with_bright_spot': run_data['stats'].images_with_bright_spot,
                'images_without_bright_spot': run_data['stats'].images_without_bright_spot,
                'images_processed_count': run_data['stats'].images_processed_count,
                'total_detections': run_data['stats'].total_detections,
                'batches_completed': run_data['stats'].batches_completed,
                'inference_times': run_data.get('inference_times', [])  # Save inference times for median calculation
            }
        }
        self.data_manager.save_checkpoint(run_id, checkpoint_data)
    
    def get_run_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a run"""
        with self.lock:
            if run_id not in self.active_runs:
                # Try to load from disk
                run_data = self.data_manager.load_run(run_id)
                if run_data:
                    return {
                        'run_id': run_id,
                        'status': run_data.status,
                        'stats': run_data.stats,
                        'progress': run_data.stats.batches_completed / run_data.stats.batches_total if run_data.stats.batches_total > 0 else 0.0
                    }
                return None
            
            run_data = self.active_runs[run_id]
            stats = run_data['stats']
            total_batches = stats.batches_total
            last_batch_completed = run_data['last_batch_completed']
            
            # Calculate progress based on completed batches and current batch progress
            batch_progress = 0.0
            current_batch_num = last_batch_completed + 1
            
            # Check if there's a batch currently being processed
            if current_batch_num <= total_batches:
                # Find the current batch
                current_batch = None
                for b in run_data['batches']:
                    if b['batch_id'] == current_batch_num:
                        current_batch = b
                        break
                
                if current_batch:
                    # Calculate progress within current batch
                    if current_batch['total_images'] > 0:
                        batch_progress = current_batch['images_processed'] / current_batch['total_images']
            
            # Overall progress: completed batches + progress in current batch
            completed_progress = last_batch_completed / total_batches if total_batches > 0 else 0.0
            current_batch_progress = (batch_progress / total_batches) if total_batches > 0 else 0.0
            progress = completed_progress + current_batch_progress
            
            # Update total images processed in stats
            total_images_processed = len(run_data['images_processed'])
            stats.total_images = run_data['total_images']  # Ensure this is set
            
            # Calculate elapsed time from start_time to now
            if stats.start_time:
                elapsed_seconds = (datetime.now() - stats.start_time).total_seconds()
                # Note: We don't store elapsed_seconds in stats, it's calculated on-the-fly
                # The frontend will calculate it from start_time
            
            return {
                'run_id': run_id,
                'status': run_data['status'],
                'stats': stats,
                'progress': min(progress, 1.0),  # Cap at 100%
                'current_batch': current_batch_num,
                'total_batches': total_batches
            }
    
    def complete_run(self, run_id: str):
        """Mark run as completed and save final data"""
        with self.lock:
            if run_id not in self.active_runs:
                return
            
            run_data = self.active_runs[run_id]
            stats = run_data['stats']
            
            # Calculate final statistics (use processed_count, not total_images)
            processed_count = stats.images_processed_count
            if processed_count > 0:
                # Success rate = percentage of images WITHOUT bright spot (good modules)
                stats.success_rate = (stats.images_without_bright_spot / processed_count * 100)
                # Bright spot rate = percentage of images WITH bright spot (defects)
                stats.bright_spot_rate = (stats.images_with_bright_spot / processed_count * 100)
            else:
                stats.success_rate = 0.0
                stats.bright_spot_rate = 0.0
            
            # Calculate final median inference time
            if run_data.get('inference_times'):
                stats.median_inference_time_seconds = statistics.median(run_data['inference_times'])
            
            stats.end_time = datetime.now()
            
            if stats.start_time and stats.end_time:
                stats.duration_seconds = (stats.end_time - stats.start_time).total_seconds()
            
            run_data['status'] = RunStatus.COMPLETED
            
            # Save final run data
            from models import RunData, BatchInfo
            # Convert batch dicts to BatchInfo objects
            batch_infos = []
            for b in run_data['batches']:
                batch_info = BatchInfo(
                    batch_id=b['batch_id'],
                    status=BatchStatus(b['status']) if isinstance(b['status'], str) else b['status'],
                    images_processed=b['images_processed'],
                    total_images=b['total_images'],
                    start_time=b.get('start_time'),
                    end_time=b.get('end_time')
                )
                batch_infos.append(batch_info)
            
            # Ensure timestamp is a datetime object
            timestamp = run_data['timestamp']
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            elif not isinstance(timestamp, datetime):
                timestamp = datetime.now()
            
            run_record = RunData(
                run_id=run_id,
                run_name=run_data['run_name'],
                timestamp=timestamp,
                model_path=run_data['model_path'],
                input_dir=run_data['input_dir'],
                output_dir=run_data['output_dir'],
                confidence=run_data['confidence'],
                batch_size=run_data['batch_size'],
                status=RunStatus.COMPLETED,
                stats=stats,
                batches=batch_infos,
                images_processed=run_data['images_processed']
            )
            
            self.data_manager.save_run(run_record)
            
            # Delete checkpoint
            self.data_manager.delete_checkpoint(run_id)
            
            # Remove from active runs
            del self.active_runs[run_id]

