import os
import torch
import cudf
import cupy as cp
from tqdm import tqdm
from collections.abc import Iterable

from nemo_curator.image.embedders.timm import TimmImageEmbedder # Inherits from ImageEmbedder
from nemo_curator.utils.distributed_utils import load_object_on_worker
from nemo_curator.utils.image.transforms import convert_transforms_to_dali # Used by parent
from crossfit.backend.cudf.series import create_list_series_from_1d_or_2d_ar # For output

# Conditional DALI import for broader script usability
try:
    from nvidia.dali import pipeline_def, fn, types
    from nvidia.dali.plugin.pytorch import feed_ndarray
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False
    # Define dummy types if DALI is not available, so the class can be imported
    class types:
        RGB = None
    def pipeline_def(*args, **kwargs): return lambda f: f
    # Define a dummy fn that can be called, but won't do anything functional
    class dummy_fn_meta(type):
        def __getattr__(cls, name):
            # This allows any attribute access on fn (e.g., fn.readers.file)
            # to return a dummy callable that does nothing or returns None.
            def dummy_callable(*args, **kwargs):
                # print(f"DALI not available: fn.{name} called with {args} {kwargs}")
                return None 
            return dummy_callable

    class fn(metaclass=dummy_fn_meta):
        pass


class PathBasedFrameEmbedder(TimmImageEmbedder):
    """
    An ImageEmbedder that loads frames from individual file paths specified in a DataFrame column,
    rather than from WebDataset TAR files. It uses a DALI pipeline for loading and transforming these frames.
    It can also run associated classifiers (like Aesthetic or NSFW) on the generated embeddings.
    """
    def __init__(self, model_name: str, frame_paths_col: str = "frame_path", **kwargs):
        """
        Args:
            model_name (str): Name of the TIMM model to use for embeddings.
            frame_paths_col (str): Column name in the input DataFrame that contains paths to frame images.
            **kwargs: Other arguments for TimmImageEmbedder (e.g., batch_size, pretrained, classifiers list).
        """
        if not DALI_AVAILABLE:
            # Raise error or print warning and disable functionality
            # For this script, we let it initialize but _run_inference will fail if DALI is called.
            # However, the prompt implies raising an error here if DALI is not found.
            raise ImportError("NVIDIA DALI is not installed. PathBasedFrameEmbedder requires DALI.")
        
        if 'classifiers' not in kwargs:
            kwargs['classifiers'] = []

        super().__init__(model_name=model_name, **kwargs)
        self.frame_paths_col = frame_paths_col
        # self.dali_transforms is initialized by TimmImageEmbedder's __init__ via self.image_transforms

    def _run_inference(
        self,
        partition_df: cudf.DataFrame,
        partition_info: dict | None = None,
    ) -> cudf.DataFrame:
        
        if not DALI_AVAILABLE: # Should have been caught in __init__, but double check
            raise RuntimeError("DALI is not available, cannot run inference.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and hasattr(self, 'device') and self.device == "cuda":
            print("Warning: CUDA device specified for embedder but torch.cuda.is_available() is False. Forcing to CPU for this worker.")
            # This might cause issues if DALI pipeline or model was strictly expecting CUDA.
            # However, DALI can run on CPU, and TIMM models can too.
        
        embedding_model = load_object_on_worker(
            self.model_name,
            self.load_embedding_model,
            {"device": device},
        )

        classifier_models = []
        for classifier_spec in self.classifiers:
            loaded_classifier_model = load_object_on_worker(
                classifier_spec.model_name, classifier_spec.load_model, {"device": device}
            )
            classifier_models.append(loaded_classifier_model)

        frame_paths_list = partition_df[self.frame_paths_col].to_pandas().tolist()
        
        # Handle empty frame_paths_list for the partition
        if not frame_paths_list:
            # Initialize empty list-dtype column for embeddings
            empty_list_col_dtype = cudf.core.dtypes.ListDtype(cp.float32(0).dtype) if hasattr(cudf.core.dtypes, 'ListDtype') else object
            partition_df[self.image_embedding_column] = cudf.Series([[] for _ in range(len(partition_df))], dtype=empty_list_col_dtype, index=partition_df.index)

            for classifier_spec in self.classifiers:
                 # Initialize with NaN or appropriate null for the classifier's pred_type
                 pred_dtype = classifier_spec.pred_type or float
                 null_val = float('nan') if pred_dtype == float else None
                 partition_df[classifier_spec.pred_column] = cudf.Series([null_val]*len(partition_df), dtype=pred_dtype, index=partition_df.index)
            return partition_df


        dali_pipe = self._build_dali_pipeline(
            frame_paths_list, 
            self.batch_size, 
            self.num_threads_per_worker, 
            0 # Assuming device_id 0 for local GPU context
        )
        dali_pipe.build()

        all_embeddings_list = []
        classifier_results_master_list = [[] for _ in self.classifiers]
        
        num_frames_in_partition = len(frame_paths_list)
        epoch_size = dali_pipe.epoch_size("FrameReader")

        processed_count = 0
        with torch.no_grad(), tqdm(
            total=epoch_size, 
            desc=f"Partition {partition_info['number'] if partition_info else 'X'} - Frames",
            disable=not self.enable_progress_bar # self.enable_progress_bar from base ImageEmbedder
        ) as pbar:
            while processed_count < epoch_size:
                try:
                    dali_output = dali_pipe.run()
                    image_batch_dali_tensor = dali_output[0]
                    
                    # Feed DALI tensor to PyTorch tensor
                    image_torch_batch = torch.empty(
                        image_batch_dali_tensor.shape(), 
                        dtype=torch.float32, 
                        device=device 
                    )
                    feed_ndarray(image_batch_dali_tensor, image_torch_batch, cuda_stream=torch.cuda.current_stream(device=device) if device == "cuda" else None)

                    current_batch_actual_size = image_torch_batch.shape[0]
                    if current_batch_actual_size == 0: break 

                    embeddings_batch = embedding_model(image_torch_batch)
                    all_embeddings_list.append(embeddings_batch.cpu())

                    for i, classifier_model in enumerate(classifier_models):
                        scores_batch = classifier_model(embeddings_batch)
                        classifier_results_master_list[i].append(scores_batch.cpu())
                    
                    pbar.update(current_batch_actual_size)
                    processed_count += current_batch_actual_size
                
                except StopIteration: break
                except Exception as e:
                    print(f"Error during DALI/model inference in partition {partition_info['number'] if partition_info else 'X'}: {e}")
                    break
        
        # Initialize columns with nulls/empty lists first
        empty_list_col_dtype = cudf.core.dtypes.ListDtype(cp.float32(0).dtype) if hasattr(cudf.core.dtypes, 'ListDtype') else object
        partition_df[self.image_embedding_column] = cudf.Series([[] for _ in range(len(partition_df))], dtype=empty_list_col_dtype, index=partition_df.index)

        for classifier_spec in self.classifiers:
            pred_dtype = classifier_spec.pred_type or float
            null_val = float('nan') if pred_dtype == float else None
            partition_df[classifier_spec.pred_column] = cudf.Series([null_val]*len(partition_df), dtype=pred_dtype, index=partition_df.index)


        if all_embeddings_list:
            final_embeddings_tensor = torch.cat(all_embeddings_list)
            
            if len(final_embeddings_tensor) != num_frames_in_partition:
                print(f"Warning: Mismatch in frame count. Expected {num_frames_in_partition}, got {len(final_embeddings_tensor)} embeddings. Padding/truncating.")
                temp_embeddings = torch.full((num_frames_in_partition, final_embeddings_tensor.shape[1]), float('nan'), device='cpu')
                min_len = min(len(final_embeddings_tensor), num_frames_in_partition)
                temp_embeddings[:min_len] = final_embeddings_tensor[:min_len]
                final_embeddings_tensor = temp_embeddings

            embeddings_cupy = cp.asarray(final_embeddings_tensor.numpy(force=True)) # force=True if tensor might be on CUDA from model
            embedding_series = create_list_series_from_1d_or_2d_ar(embeddings_cupy, index=partition_df.index)
            # Manually assign to avoid potential index mismatch if partition_df was filtered
            # This assumes the order from DALI matches the order in partition_df, which it should with shuffle_after_epoch=False
            partition_df.loc[partition_df.index[:len(embedding_series)], self.image_embedding_column] = embedding_series.values

            for i, classifier_spec in enumerate(self.classifiers):
                if classifier_results_master_list[i]:
                    final_scores_tensor = torch.cat(classifier_results_master_list[i])
                    if len(final_scores_tensor) != num_frames_in_partition:
                        temp_scores = torch.full((num_frames_in_partition,), float('nan'), device='cpu')
                        min_len_scores = min(len(final_scores_tensor), num_frames_in_partition)
                        temp_scores[:min_len_scores] = final_scores_tensor[:min_len_scores]
                        final_scores_tensor = temp_scores
                    
                    scores_cupy = cp.asarray(final_scores_tensor.numpy(force=True))
                    pred_series = cudf.Series(scores_cupy, index=partition_df.index, nan_as_null=False).astype(classifier_spec.pred_type or float)
                    partition_df.loc[partition_df.index[:len(pred_series)], classifier_spec.pred_column] = pred_series.values
                
        return partition_df

    def _build_dali_pipeline(self, frame_paths_list, batch_size, num_threads, device_id):
        # This method needs to be defined within the class or accessible to it.
        # It uses self.dali_transforms
        @pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=device_id, exec_async=True, exec_pipelined=True)
        def dali_frame_loader_pipeline():
            raw_files, _ = fn.readers.file(files=frame_paths_list, name="FrameReader", shuffle_after_epoch=False, pad_last_batch=True, dont_shuffle=True)
            imgs = fn.decoders.image(raw_files, device="mixed", output_type=types.RGB)
            
            transformed_imgs = imgs
            # self.dali_transforms is expected to be a list of DALI operations
            # It's set up by TimmImageEmbedder based on self.image_transforms
            if self.dali_transforms: # Check if dali_transforms were successfully created
                for transform_op in self.dali_transforms:
                    transformed_imgs = transform_op(transformed_imgs)
            else: # Fallback if self.dali_transforms is None or empty (e.g. only ToTensor was specified)
                  # This usually means we need a ToTensor equivalent for DALI.
                  # For TIMM models, they often expect NCHW, float, normalized.
                  # If self.dali_transforms is empty, it implies image_transforms was basic.
                  # A minimal DALI equivalent for ToTensor + Normalize:
                transformed_imgs = fn.crop_mirror_normalize(
                    transformed_imgs,
                    device="gpu" if torch.cuda.is_available() else "cpu", # Match output device
                    dtype=types.FLOAT,
                    output_layout=types.NCHW, # Timm models expect NCHW
                    # Example mean/std, replace with actual if needed or make configurable
                    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], 
                    std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
                )
            return transformed_imgs
        return dali_frame_loader_pipeline() # Instantiate the pipeline

    def load_dataset_shard(self, data_source_identifier: str) -> Iterable:
        # This method is not used by this embedder as it processes DataFrames of paths.
        raise NotImplementedError(
            "PathBasedFrameEmbedder processes DataFrames of paths directly in _run_inference, "
            "it does not load data from TAR shards using this method."
        )

```
