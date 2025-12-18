import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import uuid

from models import RunData, RunStatus, RunStats, BatchInfo


class DataManager:
    def __init__(self):
        # Get %appdata% path
        appdata_path = os.getenv('APPDATA')
        if not appdata_path:
            raise RuntimeError("APPDATA environment variable not found")
        
        self.base_dir = Path(appdata_path) / "BrightSpotDetector"
        self.runs_dir = self.base_dir / "runs"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        
        # Create directories if they don't exist
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Cleanup old runs on initialization
        self.cleanup_old_runs()
    
    def cleanup_old_runs(self):
        """Remove runs older than 1 year"""
        cutoff_date = datetime.now() - timedelta(days=365)
        deleted_count = 0
        
        for run_file in self.runs_dir.glob("*.json"):
            try:
                with open(run_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    timestamp_str = data.get('timestamp')
                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        if timestamp < cutoff_date:
                            run_file.unlink()
                            deleted_count += 1
                            # Also delete associated checkpoint if exists
                            checkpoint_file = self.checkpoints_dir / f"{run_file.stem}.json"
                            if checkpoint_file.exists():
                                checkpoint_file.unlink()
            except Exception as e:
                print(f"Error cleaning up {run_file}: {e}")
        
        if deleted_count > 0:
            print(f"Cleaned up {deleted_count} old run(s)")
    
    def generate_run_id(self) -> str:
        """Generate a unique run ID"""
        return str(uuid.uuid4())
    
    def save_run(self, run_data: RunData):
        """Save run data to disk"""
        run_file = self.runs_dir / f"{run_data.run_id}.json"
        
        # Convert to dict for JSON serialization
        # model_dump with mode='json' already converts datetime to ISO strings automatically
        data = run_data.model_dump(mode='json', exclude_none=True)
        
        with open(run_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_run(self, run_id: str) -> Optional[RunData]:
        """Load run data from disk"""
        run_file = self.runs_dir / f"{run_id}.json"
        if not run_file.exists():
            return None
        
        try:
            with open(run_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Parse datetime fields
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            if data['stats'].get('start_time'):
                data['stats']['start_time'] = datetime.fromisoformat(data['stats']['start_time'])
            if data['stats'].get('end_time'):
                data['stats']['end_time'] = datetime.fromisoformat(data['stats']['end_time'])
            
            for batch in data.get('batches', []):
                if batch.get('start_time'):
                    batch['start_time'] = datetime.fromisoformat(batch['start_time'])
                if batch.get('end_time'):
                    batch['end_time'] = datetime.fromisoformat(batch['end_time'])
            
            return RunData(**data)
        except Exception as e:
            print(f"Error loading run {run_id}: {e}")
            return None
    
    def get_all_runs(self, limit: Optional[int] = None) -> List[RunData]:
        """Get all runs, sorted by timestamp (newest first)"""
        runs = []
        for run_file in self.runs_dir.glob("*.json"):
            run_id = run_file.stem
            run_data = self.load_run(run_id)
            if run_data:
                runs.append(run_data)
        
        # Sort by timestamp descending
        runs.sort(key=lambda x: x.timestamp, reverse=True)
        
        if limit:
            runs = runs[:limit]
        
        return runs
    
    def save_checkpoint(self, run_id: str, checkpoint_data: Dict[str, Any]):
        """Save checkpoint data"""
        checkpoint_file = self.checkpoints_dir / f"{run_id}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    def load_checkpoint(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint data"""
        checkpoint_file = self.checkpoints_dir / f"{run_id}.json"
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading checkpoint {run_id}: {e}")
            return None
    
    def delete_checkpoint(self, run_id: str):
        """Delete checkpoint after successful completion"""
        checkpoint_file = self.checkpoints_dir / f"{run_id}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()

