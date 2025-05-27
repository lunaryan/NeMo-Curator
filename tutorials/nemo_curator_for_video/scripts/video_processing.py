#!/usr/bin/env python3
"""
Production Video Frame Processing Pipeline with NeMo Curator and DALI
Processes videos through frame extraction, filtering, augmentation, and recomposition
"""
import io
import os
import cv2
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import tarfile
# NeMo Curator imports
from nemo_curator.datasets.image_text_pair_dataset import ImageTextPairDataset
from nemo_curator.image.embedders import TimmImageEmbedder
from nemo_curator.image.classifiers import AestheticClassifier, NsfwClassifier
from nemo_curator import ClusteringModel, SemanticClusterLevelDedup
from nemo_curator.datasets import DocumentDataset

# DALI imports
try:
    import nvidia.dali as dali
    from nvidia.dali import pipeline_def, fn
    DALI_AVAILABLE = True
except ImportError:
    print("Warning: DALI not available. Install with: pip install nvidia-dali-cuda120")
    DALI_AVAILABLE = False

class VideoFrameProcessor:
    """Main pipeline for processing videos through frame-level filtering and augmentation"""
    
    def __init__(self, 
                 workspace_dir: str = "./video_processing_workspace",
                 aesthetic_threshold: float = 6.0,
                 nsfw_threshold: float = 0.1,
                 semantic_similarity_threshold: float = 0.9):
        """
        Initialize the video frame processing pipeline
        
        Args:
            workspace_dir: Directory for all processing artifacts
            aesthetic_threshold: Minimum aesthetic score to keep frames
            nsfw_threshold: Maximum NSFW score to keep frames  
            semantic_similarity_threshold: Cosine similarity threshold for deduplication
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)
        
        self.aesthetic_threshold = aesthetic_threshold
        self.nsfw_threshold = nsfw_threshold
        self.semantic_similarity_threshold = semantic_similarity_threshold
        
        # Setup workspace subdirectories
        self.frames_dir = self.workspace_dir / "extracted_frames"
        self.webdataset_dir = self.workspace_dir / "webdataset"
        self.filtered_dir = self.workspace_dir / "filtered_frames"
        self.augmented_dir = self.workspace_dir / "augmented_frames"
        self.output_videos_dir = self.workspace_dir / "processed_videos"
        self.metadata_dir = self.workspace_dir / "metadata"
        
        for dir_path in [self.frames_dir, self.webdataset_dir, self.filtered_dir, 
                        self.augmented_dir, self.output_videos_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
            
        # Initialize models
        self.embedding_model = None
        self.aesthetic_classifier = None
        self.nsfw_classifier = None
        
        # Video processing metadata
        self.video_metadata = {}
        
    def process_video(self, video_path: Path, video_id: str) -> Dict:
        """
        Process a single video through the complete pipeline
        
        Args:
            video_path: Path to input video
            video_id: Unique identifier for the video
            
        Returns:
            Dictionary with processing results and metadata
        """
        print(f"🎬 Processing video: {video_path.name} (ID: {video_id})")
        
        # Step 1: Extract frames to WebDataset format
        frame_metadata = self._extract_frames_to_webdataset(video_path, video_id)
        
        # Step 2: Create ImageTextPairDataset
        dataset = self._create_image_dataset(video_id)
        if not dataset:
            return None
        
        # Step 3: Generate embeddings
        dataset = self._generate_embeddings(dataset)
        
        # Step 4: Apply aesthetic filtering
        dataset = self._apply_aesthetic_filtering(dataset, video_id)
        
        # Step 5: Apply NSFW filtering  
        dataset = self._apply_nsfw_filtering(dataset, video_id)
        
        # Step 6: Apply semantic deduplication
        dataset = self._apply_semantic_deduplication(dataset, video_id)
        
        # Step 7: Apply DALI augmentation
        augmented_frames = self._apply_dali_augmentation(dataset, video_id)
        
        # Step 8: Recompose video
        output_video_path = self._recompose_video(augmented_frames, video_id, frame_metadata)
        
        # Step 9: Update metadata
        processing_results = self._update_video_metadata(
            video_id, video_path, output_video_path, frame_metadata, dataset
        )
        
        print(f"✅ Video processing complete: {output_video_path}")
        return processing_results
    
    def _extract_frames_to_webdataset(self, video_path: Path, video_id: str) -> Dict:
        """Extract video frames and organize them in WebDataset format"""
        print(f"📽️  Extracting frames for video {video_id}")
        
        video_frames_dir = self.frames_dir / video_id
        video_frames_dir.mkdir(exist_ok=True)
        
        # Extract frames using OpenCV
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_metadata = {
            'video_id': video_id,
            'original_video_path': str(video_path),
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'frames': []
        }
        
        # Create WebDataset structure
        webdataset_video_dir = self.webdataset_dir / video_id
        webdataset_video_dir.mkdir(exist_ok=True)
        
        frame_count = 0
        extracted_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Save frame as image
            frame_filename = f"{video_id}_{frame_count:06d}"
            frame_image_path = webdataset_video_dir / f"{frame_filename}.png"
            cv2.imwrite(str(frame_image_path), frame)
            
            # Create metadata entry for this frame
            frame_info = {
                'key': frame_filename,
                'frame_index': frame_count,
                'timestamp': frame_count / fps if fps > 0 else 0,
                'image_path': str(frame_image_path),
                'video_id': video_id
            }
            
            extracted_frames.append(frame_info)
            frame_count += 1
            
        cap.release()
        
        frame_metadata['frames'] = extracted_frames
        frame_metadata['extracted_count'] = len(extracted_frames)
        
        # Save frame metadata
        metadata_file = self.metadata_dir / f"{video_id}_frames.json"
        with open(metadata_file, 'w') as f:
            json.dump(frame_metadata, f, indent=2)
            
        print(f"  📸 Extracted {len(extracted_frames)} frames")
        return frame_metadata

    def _create_image_dataset_cpu(self, video_id: str) -> ImageTextPairDataset:
        """Create ImageTextPairDataset from extracted frames - CPU version"""
        print(f"📊 Creating ImageTextPairDataset for video {video_id} (CPU mode)")
        
        # Load frame metadata
        metadata_file = self.metadata_dir / f"{video_id}_frames.json"
        with open(metadata_file, 'r') as f:
            frame_metadata = json.load(f)
        
        # Create WebDataset format files
        webdataset_video_dir = self.webdataset_dir / video_id
        
        # Create simple directory structure instead of complex WebDataset
        frames_for_curator = []
        
        for idx, frame_info in enumerate(frame_metadata['frames']):
            # Copy frames to a simple structure
            simple_frame_path = webdataset_video_dir / f"frame_{idx:06d}.jpg"
            simple_frame_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy image
            import shutil
            original_path = Path(frame_info['image_path'])
            if original_path.exists():
                shutil.copy2(original_path, simple_frame_path)
                frames_for_curator.append({
                    'key': f"frame_{idx:06d}",
                    'image_path': str(simple_frame_path),
                    'frame_index': frame_info['frame_index'],
                    'video_id': video_id
                })
        
        # Create a mock dataset object that works without cuDF
        class MockImageTextPairDataset:
            def __init__(self, frame_data, path):
                self.frame_data = frame_data
                self.path = path
                self.metadata = pd.DataFrame(frame_data)
                
            def save_metadata(self):
                # Save as regular parquet using pandas
                metadata_path = Path(self.path) / "metadata.parquet"
                self.metadata.to_parquet(metadata_path, index=False)
                
        dataset = MockImageTextPairDataset(frames_for_curator, str(webdataset_video_dir))
        return dataset

    def _create_image_dataset(self, video_id: str) -> ImageTextPairDataset:
        """Create ImageTextPairDataset from extracted frames"""
        print(f"📊 Creating ImageTextPairDataset for video {video_id}")
        
        # Load frame metadata
        metadata_file = self.metadata_dir / f"{video_id}_frames.json"
        with open(metadata_file, 'r') as f:
            frame_metadata = json.load(f)
        
        # Create WebDataset format files
        webdataset_video_dir = self.webdataset_dir / video_id
        
        # Use a simple approach - create minimal parquet with just key column
        shard_id = 0
        shard_name = f"{shard_id:05d}"
        
        # Create tar file
        tar_path = webdataset_video_dir / f"{shard_name}.tar"
        keys = []
        
        with tarfile.open(tar_path, 'w') as tar:
            for idx, frame_info in enumerate(frame_metadata['frames']):
                record_id = f"{shard_id:05d}{idx:04d}"
                keys.append(record_id)
                
                # Add image to tar
                image_path = Path(frame_info['image_path'])
                if image_path.exists():
                    tar.add(image_path, arcname=f"{record_id}.jpg")
                    
                    # Add text file
                    txt_content = f"Frame {frame_info['frame_index']} from video {video_id}"
                    txt_info = tarfile.TarInfo(name=f"{record_id}.txt")
                    txt_info.size = len(txt_content.encode())
                    tar.addfile(txt_info, fileobj=io.BytesIO(txt_content.encode()))
        
        # Create minimal parquet file with just the key column
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # Create simple schema that cuDF can handle
        schema = pa.schema([
            pa.field('key', pa.string())
        ])
        
        table = pa.table([keys], schema=schema)
        parquet_path = webdataset_video_dir / f"{shard_name}.parquet"
        pq.write_table(table, parquet_path)
        
        # Create dataset
        dataset = ImageTextPairDataset.from_webdataset(
            str(webdataset_video_dir), 
            id_col="key"
        )
        
        return dataset

    def _generate_embeddings(self, dataset: ImageTextPairDataset) -> ImageTextPairDataset:
        """Generate CLIP embeddings for all frames"""
        print("🔢 Generating image embeddings...")
        
        # Setup Dask client for distributed processing
        from dask.distributed import Client, LocalCluster
        
        try:
            # Try to get existing client
            client = Client.current()
            print(f"  Using existing Dask client: {client}")
        except ValueError:
            # Create local cluster if no client exists
            print("  Creating local Dask cluster...")
            cluster = LocalCluster(
                n_workers=1,
                threads_per_worker=2,
                processes=False,  # Use threads instead of processes
                memory_limit='4GB'
            )
            client = Client(cluster)
            print(f"  Created Dask client: {client}")
        
        if self.embedding_model is None:
            self.embedding_model = TimmImageEmbedder(
                "vit_large_patch14_clip_quickgelu_224.openai",
                pretrained=True,
                batch_size=32,  # Smaller batch size
                num_threads_per_worker=2,
                normalize_embeddings=True,
                autocast=True,
            )
        
        # Generate embeddings (lazy computation)
        dataset = self.embedding_model(dataset)
        
        # Trigger computation and save
        dataset.save_metadata()
        
        print("  ✅ Embeddings generated and saved")
        return dataset

    def _apply_aesthetic_filtering(self, dataset: ImageTextPairDataset, video_id: str) -> ImageTextPairDataset:
        """Apply aesthetic quality filtering"""
        print(f"🎨 Applying aesthetic filtering (threshold: {self.aesthetic_threshold})...")
        
        if self.aesthetic_classifier is None:
            self.aesthetic_classifier = AestheticClassifier()
            
        # Apply aesthetic classification
        dataset = self.aesthetic_classifier(dataset)
        
        # Create filter column
        dataset.metadata["passes_aesthetic_check"] = (
            dataset.metadata["aesthetic_score"] > self.aesthetic_threshold
        )
        
        # Save filtered dataset
        aesthetic_output_dir = self.filtered_dir / f"{video_id}_aesthetic"
        dataset.to_webdataset(str(aesthetic_output_dir), filter_column="passes_aesthetic_check")
        
        # Update dataset to filtered version
        dataset = ImageTextPairDataset.from_webdataset(str(aesthetic_output_dir), id_col="key")
        
        aesthetic_passed = dataset.metadata["passes_aesthetic_check"].sum().compute()
        print(f"  ✅ Aesthetic filtering: {aesthetic_passed} frames passed")
        
        return dataset
    
    def _apply_nsfw_filtering(self, dataset: ImageTextPairDataset, video_id: str) -> ImageTextPairDataset:
        """Apply NSFW content filtering"""
        print(f"🔒 Applying NSFW filtering (threshold: {self.nsfw_threshold})...")
        
        if self.nsfw_classifier is None:
            self.nsfw_classifier = NsfwClassifier()
            
        # Apply NSFW classification
        dataset = self.nsfw_classifier(dataset)
        
        # Create filter column (keep frames with LOW nsfw scores)
        dataset.metadata["passes_nsfw_check"] = (
            dataset.metadata["nsfw_score"] < self.nsfw_threshold
        )
        
        # Save filtered dataset
        nsfw_output_dir = self.filtered_dir / f"{video_id}_nsfw"
        dataset.to_webdataset(str(nsfw_output_dir), filter_column="passes_nsfw_check")
        
        # Update dataset to filtered version
        dataset = ImageTextPairDataset.from_webdataset(str(nsfw_output_dir), id_col="key")
        
        nsfw_passed = dataset.metadata["passes_nsfw_check"].sum().compute()
        print(f"  ✅ NSFW filtering: {nsfw_passed} frames passed")
        
        return dataset
    
    def _apply_semantic_deduplication(self, dataset: ImageTextPairDataset, video_id: str) -> ImageTextPairDataset:
        """Apply semantic deduplication to remove similar frames"""
        print(f"🔄 Applying semantic deduplication (threshold: {self.semantic_similarity_threshold})...")
        
        # Convert to DocumentDataset for clustering
        embeddings_dataset = DocumentDataset(dataset.metadata)
        
        # Setup deduplication directories
        semantic_dedup_dir = self.filtered_dir / f"{video_id}_semantic_dedup"
        semantic_dedup_dir.mkdir(exist_ok=True)
        
        clustering_output = semantic_dedup_dir / "cluster_output"
        
        # Run clustering
        clustering_model = ClusteringModel(
            id_column="key",
            embedding_column="image_embedding",
            max_iter=10,
            n_clusters=max(1, len(dataset.metadata) // 50),  # Dynamic cluster count
            random_state=42,
            clustering_output_dir=str(clustering_output),
        )
        clustered_dataset = clustering_model(embeddings_dataset)
        
        # Run cluster-level deduplication
        emb_by_cluster_output = clustering_output / "embs_by_nearest_center"
        duplicate_output = semantic_dedup_dir / "duplicates"
        
        semantic_dedup = SemanticClusterLevelDedup(
            n_clusters=max(1, len(dataset.metadata) // 50),
            emb_by_clust_dir=str(emb_by_cluster_output),
            id_column="key",
            which_to_keep="hard",
            sim_metric="cosine", 
            embedding_column="image_embedding",
            batched_cosine_similarity=1024,
            output_dir=str(duplicate_output),
        )
        
        semantic_dedup.compute_semantic_match_dfs()
        
        # Extract duplicates with specified threshold
        eps_threshold = 1.0 - self.semantic_similarity_threshold
        deduplicated_dataset_ids = semantic_dedup.extract_dedup_data(eps_to_extract=eps_threshold)
        
        # Remove duplicates
        dataset.metadata["is_unique"] = ~dataset.metadata["key"].isin(
            deduplicated_dataset_ids.df["key"].compute()
        )
        
        deduplicated_output_dir = self.filtered_dir / f"{video_id}_deduplicated"
        dataset.to_webdataset(str(deduplicated_output_dir), filter_column="is_unique")
        
        # Update dataset to deduplicated version
        dataset = ImageTextPairDataset.from_webdataset(str(deduplicated_output_dir), id_col="key")
        
        unique_frames = dataset.metadata["is_unique"].sum().compute()
        print(f"  ✅ Semantic deduplication: {unique_frames} unique frames")
        
        return dataset
    
    def _apply_dali_augmentation(self, dataset: ImageTextPairDataset, video_id: str) -> List[Path]:
        """Apply DALI augmentation to filtered frames"""
        print("🔄 Applying DALI augmentation...")
        
        if not DALI_AVAILABLE:
            print("  ⚠️  DALI not available, copying frames without augmentation")
            return self._copy_frames_without_augmentation(dataset, video_id)
        
        # Get list of frame paths that passed all filters
        frame_paths = self._get_filtered_frame_paths(dataset, video_id)
        
        # Create augmentation pipeline
        augmented_frames = []
        augmented_video_dir = self.augmented_dir / video_id
        augmented_video_dir.mkdir(exist_ok=True)
        
        @pipeline_def
        def augmentation_pipeline():
            # Read images
            images = fn.readers.file(files=frame_paths)
            images = fn.decoders.image(images, device="mixed")
            
            # Apply augmentations suitable for medical/training data
            # Geometric augmentations
            images = fn.resize(images, size=[224, 224])
            images = fn.rotate(images, angle=fn.random.uniform(range=[-10, 10]))
            
            # Color augmentations (mild for medical data)
            images = fn.brightness_contrast(
                images,
                brightness=fn.random.uniform(range=[0.9, 1.1]),
                contrast=fn.random.uniform(range=[0.9, 1.1])
            )
            
            # Noise augmentation
            images = fn.noise.gaussian(images, stddev=fn.random.uniform(range=[0, 5]))
            
            return images
        
        # Create and build pipeline
        pipe = augmentation_pipeline(batch_size=8, num_threads=4, device_id=0)
        pipe.build()
        
        # Process frames in batches
        batch_count = 0
        for batch_idx in range((len(frame_paths) + 7) // 8):  # Ceiling division
            outputs = pipe.run()
            batch_images = outputs[0].as_cpu().as_array()
            
            # Save augmented frames
            for i, aug_image in enumerate(batch_images):
                frame_idx = batch_idx * 8 + i
                if frame_idx < len(frame_paths):
                    aug_frame_path = augmented_video_dir / f"aug_frame_{frame_idx:06d}.jpg"
                    
                    # Convert DALI output to OpenCV format and save
                    # DALI outputs CHW format, OpenCV expects HWC
                    if len(aug_image.shape) == 3:
                        aug_image = np.transpose(aug_image, (1, 2, 0))
                    
                    cv2.imwrite(str(aug_frame_path), aug_image)
                    augmented_frames.append(aug_frame_path)
            
            batch_count += 1
        
        print(f"  ✅ DALI augmentation: {len(augmented_frames)} frames augmented")
        return augmented_frames
    
    def _copy_frames_without_augmentation(self, dataset: ImageTextPairDataset, video_id: str) -> List[Path]:
        """Fallback method when DALI is not available"""
        frame_paths = self._get_filtered_frame_paths(dataset, video_id)
        
        augmented_video_dir = self.augmented_dir / video_id
        augmented_video_dir.mkdir(exist_ok=True)
        
        copied_frames = []
        for i, frame_path in enumerate(frame_paths):
            dest_path = augmented_video_dir / f"frame_{i:06d}.jpg"
            shutil.copy2(frame_path, dest_path)
            copied_frames.append(dest_path)
            
        return copied_frames
    
    def _get_filtered_frame_paths(self, dataset: ImageTextPairDataset, video_id: str) -> List[Path]:
        """Extract frame paths from filtered dataset"""
        frame_paths = []
        
        # Get the keys from the filtered dataset
        try:
            keys = dataset.metadata['key'].compute().tolist()
            
            # Reconstruct paths from the WebDataset structure
            webdataset_video_dir = self.webdataset_dir / video_id
            
            for key in keys:
                # Extract from tar file using the key
                shard_id = key[:5]  # First 5 digits are shard ID
                tar_path = webdataset_video_dir / f"{shard_id}.tar"
                
                if tar_path.exists():
                    # Extract image from tar to temporary location
                    temp_image_path = self.augmented_dir / video_id / f"temp_{key}.jpg"
                    temp_image_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with tarfile.open(tar_path, 'r') as tar:
                        try:
                            member = tar.getmember(f"{key}.jpg")
                            tar.extract(member, path=temp_image_path.parent)
                            # Rename to expected path
                            extracted_path = temp_image_path.parent / f"{key}.jpg"
                            if extracted_path.exists():
                                extracted_path.rename(temp_image_path)
                                frame_paths.append(temp_image_path)
                        except KeyError:
                            print(f"Warning: {key}.jpg not found in {tar_path}")
            
        except Exception as e:
            print(f"Warning: Could not extract frame paths from dataset: {e}")
            # Fallback: use original extracted frames
            original_frames_dir = self.frames_dir / video_id
            if original_frames_dir.exists():
                frame_paths = list(original_frames_dir.glob("*.jpg"))[:10]  # Limit for safety
        
        return sorted(frame_paths)
    
    def _recompose_video(self, augmented_frames: List[Path], video_id: str, original_metadata: Dict) -> Path:
        """Recompose augmented frames back into a video"""
        print(f"🎬 Recomposing video from {len(augmented_frames)} augmented frames...")
        
        if not augmented_frames:
            print("  ⚠️  No frames to recompose")
            return None
            
        output_video_path = self.output_videos_dir / f"{video_id}_processed.mp4"
        
        # Get frame dimensions from first frame
        first_frame = cv2.imread(str(augmented_frames[0]))
        height, width = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = original_metadata.get('fps', 30.0)
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        # Write frames to video
        for frame_path in sorted(augmented_frames):
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                # Resize frame if needed
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                out.write(frame)
        
        out.release()
        
        print(f"  ✅ Video recomposed: {output_video_path}")
        return output_video_path
    
    def _update_video_metadata(self, video_id: str, original_path: Path, 
                              processed_path: Path, frame_metadata: Dict,
                              final_dataset: ImageTextPairDataset) -> Dict:
        """Update and save comprehensive video processing metadata"""
        
        # Calculate processing statistics
        original_frame_count = frame_metadata['extracted_count']
        final_frame_count = len(final_dataset.metadata) if final_dataset else 0
        
        processing_results = {
            'video_id': video_id,
            'original_video_path': str(original_path),
            'processed_video_path': str(processed_path) if processed_path else None,
            'original_frame_count': original_frame_count,
            'final_frame_count': final_frame_count,
            'retention_rate': final_frame_count / original_frame_count if original_frame_count > 0 else 0,
            'processing_metadata': {
                'aesthetic_threshold': self.aesthetic_threshold,
                'nsfw_threshold': self.nsfw_threshold,
                'semantic_similarity_threshold': self.semantic_similarity_threshold,
                'original_fps': frame_metadata.get('fps', 0),
                'original_dimensions': {
                    'width': frame_metadata.get('width', 0),
                    'height': frame_metadata.get('height', 0)
                }
            },
            'workspace_paths': {
                'frames_dir': str(self.frames_dir / video_id),
                'webdataset_dir': str(self.webdataset_dir / video_id),
                'filtered_dir': str(self.filtered_dir),
                'augmented_dir': str(self.augmented_dir / video_id),
            }
        }
        
        # Save individual video metadata
        video_metadata_file = self.metadata_dir / f"{video_id}_processing_results.json"
        with open(video_metadata_file, 'w') as f:
            json.dump(processing_results, f, indent=2)
            
        # Update global metadata
        self.video_metadata[video_id] = processing_results
        
        # Save global metadata
        global_metadata_file = self.metadata_dir / "all_videos_metadata.json"
        with open(global_metadata_file, 'w') as f:
            json.dump(self.video_metadata, f, indent=2)
            
        return processing_results
    
    def process_video_batch(self, video_paths: List[Path]) -> Dict:
        """Process multiple videos through the pipeline"""
        print(f"🎥 Processing batch of {len(video_paths)} videos")
        
        batch_results = {}
        
        for i, video_path in enumerate(video_paths):
            video_id = f"video_{i:04d}_{video_path.stem}"
            
            try:
                results = self.process_video(video_path, video_id)
                batch_results[video_id] = results
                print(f"  ✅ Completed {i+1}/{len(video_paths)}: {video_path.name}")
                
            except Exception as e:
                print(f"  ❌ Failed to process {video_path.name}: {e}")
                batch_results[video_id] = {'error': str(e)}
                
        # Save batch summary
        batch_summary_file = self.metadata_dir / "batch_processing_summary.json"
        with open(batch_summary_file, 'w') as f:
            json.dump(batch_results, f, indent=2)
            
        print(f"🎯 Batch processing complete. Results saved to {batch_summary_file}")
        return batch_results
    
    def get_processing_summary(self) -> Dict:
        """Get summary statistics of all processed videos"""
        if not self.video_metadata:
            return {"message": "No videos processed yet"}
            
        total_videos = len(self.video_metadata)
        total_original_frames = sum(v.get('original_frame_count', 0) for v in self.video_metadata.values())
        total_final_frames = sum(v.get('final_frame_count', 0) for v in self.video_metadata.values())
        
        summary = {
            'total_videos_processed': total_videos,
            'total_original_frames': total_original_frames,
            'total_final_frames': total_final_frames,
            'overall_retention_rate': total_final_frames / total_original_frames if total_original_frames > 0 else 0,
            'average_retention_rate': np.mean([v.get('retention_rate', 0) for v in self.video_metadata.values()]),
            'processing_settings': {
                'aesthetic_threshold': self.aesthetic_threshold,
                'nsfw_threshold': self.nsfw_threshold,
                'semantic_similarity_threshold': self.semantic_similarity_threshold
            }
        }
        
        return summary

# Example usage and testing
def main():
    """Example usage of the VideoFrameProcessor"""
    
    # Initialize processor
    processor = VideoFrameProcessor(
        workspace_dir="./video_processing_workspace",
        aesthetic_threshold=6.0,
        nsfw_threshold=0.1,
        semantic_similarity_threshold=0.9
    )
    
    # Example: Process a single video
    video_path = Path("/data4/user/yan390/curator/NeMo-Curator/Suturing/video/Suturing_B001_capture1.avi")
    if video_path.exists():
        results = processor.process_video(video_path, "test_video_001")
        print("\n📊 Processing Results:")
        print(json.dumps(results, indent=2))
    exit(0) 
    # Example: Process multiple videos
    video_directory = Path("path/to/video/directory")
    if video_directory.exists():
        video_files = list(video_directory.glob("*.mp4"))
        if video_files:
            batch_results = processor.process_video_batch(video_files[:5])  # Process first 5
            
            # Get summary
            summary = processor.get_processing_summary()
            print("\n📈 Processing Summary:")
            print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()