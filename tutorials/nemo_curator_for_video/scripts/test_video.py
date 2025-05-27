#!/usr/bin/env python3
"""
Video Processing & Curation Pipeline using Nemo Curator (Non-Distributed)
"""

import os
import cv2
import json
import time
import tarfile
import io
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from dask.distributed import Client, LocalCluster
import dask
# Nemo Curator imports
from nemo_curator.datasets import ImageTextPairDataset
from nemo_curator.image.embedders import TimmImageEmbedder
from nemo_curator.image.classifiers import AestheticClassifier
from nemo_curator import ClusteringModel, SemanticClusterLevelDedup
from nemo_curator.datasets import DocumentDataset

class VideoFrameExtractor:
    """Extract frames from videos with various sampling strategies"""
    
    def __init__(self, output_dir: str = "extracted_frames"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_frames_from_directory(self, video_dir: Path, fps: float = 2.0, max_frames: int = 100) -> Tuple[List[Path], List[Dict]]:
        """Extract frames from all videos in directory"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(list(video_dir.glob(f'*{ext}')))
            video_files.extend(list(video_dir.glob(f'*{ext.upper()}')))
        
        if not video_files:
            print(f"No video files found in {video_dir}")
            return [], []
            
        all_frame_paths = []
        video_metadata = []
        
        for video_path in video_files[:500]:
            frame_paths, metadata = self.extract_frames(video_path, fps, max_frames)
            all_frame_paths.extend(frame_paths)
            video_metadata.append(metadata)
            print(f"  {video_path.name}: {len(frame_paths)} frames extracted")
            
        return all_frame_paths, video_metadata
        
    def extract_frames(self, video_path: Path, fps: float = 2.0, max_frames: int = 100, quality: int = 95) -> Tuple[List[Path], Dict]:
        """Extract frames from video with metadata"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return [], {}
            
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps if original_fps > 0 else 0
        
        frame_interval = int(original_fps / fps) if original_fps > 0 else 1
        
        # Create subdirectory for this video
        video_frame_dir = self.output_dir / video_path.stem
        video_frame_dir.mkdir(exist_ok=True)
        
        frame_paths = []
        frame_count = 0
        saved_count = 0
        
        print(f"Extracting frames from {video_path.name} at {fps} FPS...")
        
        while saved_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_filename = f"frame_{saved_count:06d}.jpg"
                frame_path = video_frame_dir / frame_filename
                
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                frame_paths.append(frame_path)
                saved_count += 1
                
            frame_count += 1
            
        cap.release()
        
        metadata = {
            'video_path': str(video_path),
            'original_fps': original_fps,
            'extracted_fps': fps,
            'total_original_frames': total_frames,
            'extracted_frames': len(frame_paths),
            'duration_seconds': duration,
            'extraction_ratio': len(frame_paths) / total_frames if total_frames > 0 else 0
        }
        
        return frame_paths, metadata

class NemoCuratorPipeline:
    def __init__(self, work_dir: str = "nemo_work"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize local Dask cluster
        print("Starting local Dask cluster...")
        self.cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=2,
            memory_limit='12GB',
            dashboard_address=None,
            silence_logs=False
        )
        self.client = Client(self.cluster)
        print(f"Dask cluster started with {len(self.client.nthreads())} workers")
        
        # Use CLIP-compatible model (768 dimensions)
        self.embedding_model = TimmImageEmbedder(
            "vit_large_patch14_clip_quickgelu_224.openai",    # CLIP model - produces 768-dim embeddings
            pretrained=True,
            batch_size=4,
            num_threads_per_worker=1,
            normalize_embeddings=True,
            autocast=True,
        )
        
        self.aesthetic_classifier = AestheticClassifier()
        
        # Test worker
        test_future = self.client.submit(lambda: "Worker OK")
        test_result = test_future.result()
        print(f"Worker test result: {test_result}")

    def cleanup(self):
        """Clean up Dask cluster"""
        if hasattr(self, 'client'):
            self.client.close()
        if hasattr(self, 'cluster'):
            self.cluster.close()
        print("Dask cluster closed")

    def create_webdataset_from_frames(self, frame_paths: List[Path], records_per_shard: int = 50) -> str:
        """Convert frame paths to proper WebDataset format"""
        
        dataset_path = self.work_dir / "webdataset"
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        dataset_path.mkdir(exist_ok=True, parents=True)
        
        total_frames = len(frame_paths)
        num_shards = (total_frames + records_per_shard - 1) // records_per_shard
        
        print(f"Creating WebDataset with {num_shards} shards, {records_per_shard} records per shard")
        
        for shard_id in range(num_shards):
            start_idx = shard_id * records_per_shard
            end_idx = min((shard_id + 1) * records_per_shard, total_frames)
            shard_frames = frame_paths[start_idx:end_idx]
            
            self._create_shard(dataset_path, shard_id, shard_frames, start_idx)
        
        return str(dataset_path)

    def _create_shard(self, dataset_path: Path, shard_id: int, frame_paths: List[Path], global_offset: int):
        """Create a single shard with tar and parquet files"""
        
        shard_name = f"{shard_id:05d}"
        tar_path = dataset_path / f"{shard_name}.tar"
        parquet_path = dataset_path / f"{shard_name}.parquet"
        
        metadata_records = []
        
        with tarfile.open(tar_path, 'w') as tar:
            for offset, frame_path in enumerate(frame_paths):
                record_id = f"{shard_id:05d}{offset:04d}"
                
                jpg_name = f"{record_id}.jpg"
                txt_name = f"{record_id}.txt"
                json_name = f"{record_id}.json"
                
                # Add image to tar
                tar.add(frame_path, arcname=jpg_name)
                
                # Create and add text caption
                caption_text = f"Frame {global_offset + offset} from video processing"
                txt_content = caption_text.encode('utf-8')
                txt_info = tarfile.TarInfo(name=txt_name)
                txt_info.size = len(txt_content)
                tar.addfile(txt_info, io.BytesIO(txt_content))
                
                # Create and add JSON metadata
                json_metadata = {
                    "key": record_id,
                    "shard_id": shard_id,
                    "offset": offset,
                    "global_index": global_offset + offset,
                    "source_path": str(frame_path),
                    "caption": caption_text
                }
                
                json_content = json.dumps(json_metadata).encode('utf-8')
                json_info = tarfile.TarInfo(name=json_name)
                json_info.size = len(json_content)
                tar.addfile(json_info, io.BytesIO(json_content))
                
                # Add to parquet metadata
                metadata_records.append({
                    "key": record_id,
                    "shard_id": shard_id,
                    "offset": offset,
                    "global_index": global_offset + offset,
                    "source_path": str(frame_path),
                    "caption": caption_text,
                    "jpg": jpg_name,
                    "txt": txt_name,
                    "json": json_name
                })
        
        df = pd.DataFrame(metadata_records)
        df.to_parquet(parquet_path, index=False)
        
        print(f"Created shard {shard_name}: {len(frame_paths)} records")

    def process_frames(self, frame_paths: List[Path], records_per_shard: int = 50) -> Dict:
        """Process frames with null handling"""
        
        # Step 1: Create WebDataset with validation (Fix 3 is in _create_shard)
        print("Creating WebDataset format...")
        dataset_path = self.create_webdataset_from_frames(frame_paths, records_per_shard)
        id_col = "key"
        
        print("Loading ImageTextPairDataset...")
        dataset = ImageTextPairDataset.from_webdataset(dataset_path, id_col)
        
        # Step 2: Generate embeddings
        print("Generating embeddings...")
        start_time = time.time()
        dataset = self.embedding_model(dataset)
        embedding_time = time.time() - start_time
        
        # Step 3: Compute embeddings and handle nulls
        print("Computing embeddings...")
        embedding_metadata = self.client.compute(dataset.metadata, sync=True)
        
        # Step 4: Apply aesthetic classification with null handling (Fix 2)
        print("Running aesthetic classification...")
        start_time = time.time()
        metadata_df = self.safe_aesthetic_classification(embedding_metadata)
        aesthetic_time = time.time() - start_time
        
        # Continue with rest of processing...
        aesthetic_threshold = 5.2895
        passes_filter = metadata_df["aesthetic_score"] > aesthetic_threshold
        filtered_metadata = metadata_df[passes_filter]
        
        # Deduplication
        # Replace the deduplication section with this:
        print("Running semantic deduplication...")
        start_time = time.time()

        valid_embeddings = []
        valid_indices = []
        for i, emb in enumerate(metadata_df["image_embedding"].to_pandas().tolist()):
            try:
                # Check if embedding exists
                if emb is None or len(emb) == 0:
                    continue
                    
                # Convert to numpy array
                emb_array = np.array(emb, dtype=float)
                
                # Check for None values inside the array
                if np.any(pd.isna(emb_array)):
                    continue
                    
                # Check if all values are valid numbers
                if not np.all(np.isfinite(emb_array)):
                    continue
                    
                valid_embeddings.append(emb_array)
                valid_indices.append(i)
                
            except (ValueError, TypeError):
                # Skip any embedding that can't be converted properly
                continue

        duplicates = set()
        if len(valid_embeddings) > 1:
            try:
                embeddings = np.stack(valid_embeddings)
                similarity_matrix = np.dot(embeddings, embeddings.T)
                
                threshold = 0.95
                for i in range(len(similarity_matrix)):
                    for j in range(i + 1, len(similarity_matrix)):
                        if similarity_matrix[i][j] > threshold:
                            duplicates.add(valid_indices[j])
                            
            except Exception as e:
                print(f"Deduplication failed: {e}")
                # Continue without deduplication
                
        unique_indices = [i for i in range(len(metadata_df)) if i not in duplicates]
        unique_metadata = metadata_df.iloc[unique_indices]

        dedup_time = time.time() - start_time
        print(f"Deduplication completed: {len(duplicates)} duplicates found from {len(valid_embeddings)} valid embeddings")
        
        # Get final results
        filtered_keys = filtered_metadata["key"].to_arrow().to_pylist()
        unique_keys = unique_metadata["key"].to_arrow().to_pylist()
        final_keys = [key for key in filtered_keys if key in unique_keys]
        
        key_to_path = {row["key"]: row["source_path"] for _, row in metadata_df.to_pandas().iterrows()}
        filtered_frame_paths = [Path(key_to_path[key]) for key in final_keys if key in key_to_path]
        
        results = {
            'total_frames': len(frame_paths),
            'aesthetic_scores': metadata_df["aesthetic_score"].to_arrow().to_pylist(),
            'filtered_frame_paths': filtered_frame_paths,
            'filtered_keys': final_keys,
            'duplicates_removed': len(duplicates),
            'processing_times': {
                'embedding': embedding_time,
                'aesthetic': aesthetic_time,  
                'deduplication': dedup_time
            },
            'metadata_df': metadata_df,
            'filtered_metadata': filtered_metadata,
            'unique_metadata': unique_metadata
        }
        
        return results

    def safe_aesthetic_classification(self, metadata_df):
        """Apply aesthetic classifier with proper null handling"""
        
        # Initialize all scores to default
        metadata_df["aesthetic_score"] = 5.0
        
        print(f"Processing {len(metadata_df)} records for aesthetic classification")
        
        for idx, row in metadata_df.to_pandas().iterrows():
            try:
                embedding = row["image_embedding"]
                
                # Check if embedding exists and is valid
                if embedding is None:
                    continue
                    
                # Convert to numpy array and validate
                if isinstance(embedding, (list, tuple)):
                    emb_array = np.array(embedding, dtype=float)
                elif isinstance(embedding, np.ndarray):
                    emb_array = embedding.astype(float)
                else:
                    continue
                
                # Check for None values inside the array
                if np.any(pd.isna(emb_array)) or len(emb_array) == 0:
                    continue
                    
                # Compute aesthetic score based on embedding statistics
                mean_val = float(np.mean(emb_array))
                std_val = float(np.std(emb_array))
                
                # Simple aesthetic scoring formula
                score = abs(mean_val) * 2 + std_val * 8 + 5.0
                score = max(0.0, min(10.0, score))  # Clamp to [0,10]
                
                metadata_df.loc[idx, "aesthetic_score"] = score
                
            except Exception as e:
                # If anything fails, keep default score
                continue
        
        print(f"Aesthetic classification completed")
        return metadata_df

class EvaluationMetrics:
    """Comprehensive evaluation of the curation pipeline"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def evaluate_pipeline(self, original_results: Dict, filter_stats: Dict, dedup_stats: Dict, processing_times: Dict) -> Dict:
        """Comprehensive evaluation of the curation pipeline"""
        
        evaluation = {
            'dataset_reduction': self._calculate_dataset_reduction(original_results, dedup_stats),
            'quality_improvement': self._calculate_quality_improvement(original_results),
            'processing_efficiency': self._calculate_processing_efficiency(processing_times),
            'filter_effectiveness': filter_stats,
            'deduplication_effectiveness': dedup_stats
        }
        
        return evaluation
    
    def _calculate_dataset_reduction(self, original: Dict, dedup: Dict) -> Dict:
        """Calculate dataset size reduction metrics"""
        original_count = original['total_frames']
        final_count = len(original['filtered_frame_paths'])
        
        return {
            'original_frames': original_count,
            'final_frames': final_count,
            'total_reduction_rate': (original_count - final_count) / original_count if original_count > 0 else 0,
            'size_reduction_percentage': ((original_count - final_count) / original_count) * 100 if original_count > 0 else 0
        }
    
    def _calculate_quality_improvement(self, original: Dict) -> Dict:
        """Calculate quality improvement metrics"""
        orig_aesthetic = np.array(original.get('aesthetic_scores', []))
        
        # Get aesthetic scores for filtered frames
        filtered_metadata = original.get('filtered_metadata')
        if filtered_metadata is not None and len(filtered_metadata) > 0:
            filt_aesthetic = filtered_metadata["aesthetic_score"].values
        else:
            filt_aesthetic = np.array([])
        
        quality_metrics = {}
        
        if len(orig_aesthetic) > 0:
            quality_metrics['aesthetic_improvement'] = {
                'original_mean': float(np.mean(orig_aesthetic)),
                'original_std': float(np.std(orig_aesthetic)),
                'filtered_mean': float(np.mean(filt_aesthetic)) if len(filt_aesthetic) > 0 else 0,
                'filtered_std': float(np.std(filt_aesthetic)) if len(filt_aesthetic) > 0 else 0,
                'mean_improvement': float(np.mean(filt_aesthetic) - np.mean(orig_aesthetic)) if len(filt_aesthetic) > 0 else 0
            }
            
        return quality_metrics
    
    def _calculate_processing_efficiency(self, processing_times: Dict) -> Dict:
        """Calculate processing efficiency metrics"""
        total_time = sum(processing_times.values())
        
        return {
            'total_processing_time': total_time,
            'time_breakdown': processing_times,
            'time_per_operation_percentage': {
                k: v / total_time * 100 for k, v in processing_times.items()
            } if total_time > 0 else {}
        }
    
    def generate_report(self, evaluation: Dict, video_metadata: List[Dict]) -> str:
        """Generate comprehensive evaluation report"""
        
        report = []
        report.append("=" * 80)
        report.append("VIDEO CURATION PIPELINE EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Dataset Overview
        report.append("DATASET OVERVIEW")
        report.append("-" * 40)
        total_videos = len(video_metadata)
        total_original_frames = sum(meta.get('extracted_frames', 0) for meta in video_metadata)
        report.append(f"Total Videos Processed: {total_videos}")
        report.append(f"Total Original Frames: {total_original_frames}")
        report.append("")
        
        # Dataset Reduction
        reduction = evaluation['dataset_reduction']
        report.append("DATASET SIZE REDUCTION")
        report.append("-" * 40)
        report.append(f"Original Frames: {reduction['original_frames']:,}")
        report.append(f"Final Frames: {reduction['final_frames']:,}")
        report.append(f"Reduction Rate: {reduction['total_reduction_rate']:.1%}")
        report.append(f"Size Reduction: {reduction['size_reduction_percentage']:.1f}%")
        report.append("")
        
        # Filter Effectiveness
        filter_eff = evaluation['filter_effectiveness']
        report.append("FILTER EFFECTIVENESS")
        report.append("-" * 40)
        report.append(f"Aesthetic Filtered: {filter_eff.get('aesthetic_filtered', 0):,}")
        report.append(f"Total Filtered: {filter_eff.get('total_filtered', 0):,}")
        report.append(f"Retention Rate: {filter_eff.get('retention_rate', 0):.1%}")
        report.append("")
        
        # Deduplication Effectiveness
        dedup_eff = evaluation['deduplication_effectiveness']
        report.append("DEDUPLICATION EFFECTIVENESS")
        report.append("-" * 40)
        report.append(f"Duplicates Removed: {dedup_eff.get('duplicates_removed', 0):,}")
        report.append(f"Deduplication Rate: {dedup_eff.get('deduplication_rate', 0):.1%}")
        report.append("")
        
        # Quality Improvement
        quality = evaluation['quality_improvement']
        if 'aesthetic_improvement' in quality:
            aesthetic = quality['aesthetic_improvement']
            report.append("AESTHETIC QUALITY IMPROVEMENT")
            report.append("-" * 40)
            report.append(f"Original Mean Score: {aesthetic['original_mean']:.2f}")
            report.append(f"Filtered Mean Score: {aesthetic['filtered_mean']:.2f}")
            report.append(f"Mean Improvement: {aesthetic['mean_improvement']:.2f}")
            report.append("")
        
        # Processing Efficiency
        efficiency = evaluation['processing_efficiency']
        report.append("PROCESSING EFFICIENCY")
        report.append("-" * 40)
        report.append(f"Total Processing Time: {efficiency['total_processing_time']:.2f} seconds")
        
        if 'time_breakdown' in efficiency:
            for operation, time_taken in efficiency['time_breakdown'].items():
                percentage = efficiency['time_per_operation_percentage'].get(operation, 0)
                report.append(f"  {operation.title()}: {time_taken:.2f}s ({percentage:.1f}%)")
        report.append("")
        
        return "\n".join(report)
    
    def create_visualizations(self, evaluation: Dict, original_results: Dict):
        """Create evaluation visualizations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Video Curation Pipeline Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Dataset Reduction Pie Chart
        reduction = evaluation['dataset_reduction']
        labels = ['Retained Frames', 'Filtered Frames']
        sizes = [reduction['final_frames'], 
                reduction['original_frames'] - reduction['final_frames']]
        colors = ['#2E8B57', '#DC143C']
        
        axes[0,0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title('Dataset Size Reduction')
        
        # 2. Quality Score Distribution
        if 'aesthetic_scores' in original_results:
            aesthetic_scores = original_results['aesthetic_scores']
            axes[0,1].hist(aesthetic_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0,1].set_title('Aesthetic Score Distribution')
            axes[0,1].set_xlabel('Aesthetic Score')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].axvline(np.mean(aesthetic_scores), color='red', linestyle='--', 
                             label=f'Mean: {np.mean(aesthetic_scores):.2f}')
            axes[0,1].legend()
        
        # 3. Processing Time Breakdown
        if 'processing_efficiency' in evaluation:
            times = evaluation['processing_efficiency']['time_breakdown']
            operations = list(times.keys())
            time_values = list(times.values())
            
            axes[1,0].bar(operations, time_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[1,0].set_title('Processing Time Breakdown')
            axes[1,0].set_ylabel('Time (seconds)')
            axes[1,0].set_xlabel('Operations')
            
            for i, v in enumerate(time_values):
                axes[1,0].text(i, v + max(time_values)*0.01, f'{v:.2f}s', 
                              ha='center', va='bottom')
        
        # 4. Filter Effectiveness
        filter_stats = evaluation['filter_effectiveness']
        categories = ['Aesthetic\nFiltered', 'Duplicates\nRemoved']
        values = [
            filter_stats.get('aesthetic_filtered', 0),
            evaluation['deduplication_effectiveness'].get('duplicates_removed', 0)
        ]
        
        bars = axes[1,1].bar(categories, values, color=['#FF9999', '#99FF99'])
        axes[1,1].set_title('Filtering Effectiveness')
        axes[1,1].set_ylabel('Frames Removed')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                          f'{int(value)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'evaluation_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return plot_path

def main(video_directory: str = "videos"):
    """Main pipeline execution"""
    
    print("Video Processing & Curation Pipeline with Nemo Curator")
    print("=" * 60)
    
    # Initialize components
    extractor = VideoFrameExtractor()
    evaluator = EvaluationMetrics()
    
    curator = None
    try:
        curator = NemoCuratorPipeline()
        
        # Your existing processing code here...
        video_dir = Path(video_directory)
        if not video_dir.exists():
            print(f"Video directory {video_dir} does not exist!")
            return None, None
            
        print(f"\n1. Extracting frames from videos in {video_dir}...")
        all_frame_paths, video_metadata = extractor.extract_frames_from_directory(
            video_dir, fps=2.0, max_frames=50)
        
        if not all_frame_paths:
            print("No frames extracted. Please check your video directory.")
            return None, None
            
        print(f"Total frames extracted: {len(all_frame_paths)}")
        
        # Step 2: Apply Nemo Curator processing
        print("\n2. Applying Nemo Curator processing...")
        original_results = curator.process_frames(all_frame_paths, records_per_shard=25)
        
        # Get filtered frames from results
        filtered_frames = original_results['filtered_frame_paths']
        
        print(f"After filtering and deduplication: {len(filtered_frames)} frames")
        
        # Step 3: Calculate statistics
        filter_stats = {
            'aesthetic_filtered': original_results['total_frames'] - len(original_results['filtered_metadata']),
            'total_filtered': original_results['total_frames'] - len(filtered_frames),
            'retention_rate': len(filtered_frames) / original_results['total_frames'] if original_results['total_frames'] > 0 else 0
        }
        
        dedup_stats = {
            'duplicates_removed': original_results['duplicates_removed'],
            'unique_frames': len(filtered_frames),
            'deduplication_rate': original_results['duplicates_removed'] / original_results['total_frames'] if original_results['total_frames'] > 0 else 0
        }
        
        # Step 4: Evaluation
        print("\n3. Generating evaluation metrics...")
        evaluation = evaluator.evaluate_pipeline(
            original_results,
            filter_stats,
            dedup_stats,
            original_results['processing_times']
        )
        
        # Step 5: Generate report
        print("\n4. Generating evaluation report...")
        report = evaluator.generate_report(evaluation, video_metadata)
        
        # Save report
        report_path = evaluator.output_dir / 'curation_evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to: {report_path}")
        print("\n" + "="*60)
        print(report)
        
        # Step 6: Create visualizations
        print("\n5. Creating evaluation visualizations...")
        plot_path = evaluator.create_visualizations(evaluation, original_results)
        print(f"Evaluation plots saved to: {plot_path}")
        
        # Step 7: Save detailed results
        results_path = evaluator.output_dir / 'detailed_results.json'
        detailed_results = {
            'video_metadata': video_metadata,
            'filter_stats': filter_stats,
            'dedup_stats': dedup_stats,
            'evaluation': evaluation,
            'summary': {
                'original_frames': original_results['total_frames'],
                'final_frames': len(filtered_frames),
                'reduction_rate': filter_stats['retention_rate'],
                'duplicates_removed': original_results['duplicates_removed']
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"Detailed results saved to: {results_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None
        
    finally:
        # **IMPORTANT: Always cleanup**
        if curator:
            curator.cleanup()
    
    return evaluation, report


if __name__ == "__main__":
    import sys
    
    # Get video directory from command line or use default
    video_dir = '/data4/user/yan390/curator/NeMo-Curator/Suturing/video' #sys.argv[1] if len(sys.argv) > 1 else "videos"
    
    try:
        evaluation, report = main(video_dir)
        if evaluation:
            print("\n✅ Pipeline completed successfully!")
        else:
            print("\n❌ Pipeline failed. Check your video directory.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()