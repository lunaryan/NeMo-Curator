import pandas as pd
import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity # Can be used if embeddings are not pre-normalized

def _calculate_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """ Calculates cosine similarity between two 1D numpy arrays (embeddings).
        Assumes embeddings are already L2 normalized for efficiency if using dot product.
    """
    # If not normalized:
    # sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    # If normalized:
    sim = np.dot(emb1, emb2)
    return float(sim)

def deduplicate_video_frames_sequential(
    frame_df: pd.DataFrame, 
    embedding_col: str = "image_embedding",
    similarity_threshold: float = 0.98,
    min_sequence_break: int = 1 # Number of dissimilar frames to reset "previous" frame
) -> pd.DataFrame:
    """
    Identifies and flags sequential duplicate frames within a single video's frame list.
    Keeps the first frame of a duplicate sequence.

    Args:
        frame_df (pd.DataFrame): DataFrame containing frame information for a SINGLE video,
                                 including an 'embedding_col' with image embeddings (e.g., NumPy arrays or lists).
                                 Assumes frames are already sorted chronologically.
        embedding_col (str): Name of the column holding the frame embeddings.
        similarity_threshold (float): Cosine similarity threshold to consider frames duplicates.
        min_sequence_break (int): How many consecutive frames must be dissimilar to the original
                                   reference of a sequence before a new frame can be considered a
                                   new reference. E.g., if 1, then frame K being different from frame J
                                   (where J was the start of a duplicate sequence) allows K+1 to be a new reference.

    Returns:
        pd.DataFrame: The input DataFrame with an added 'is_intra_video_duplicate' boolean column.
                      True indicates the frame is a duplicate to be removed.
    """
    if frame_df.empty:
        frame_df['is_intra_video_duplicate'] = pd.Series(dtype=bool)
        return frame_df

    # Ensure embeddings are NumPy arrays for dot product
    # Convert list of embeddings to a 2D NumPy array if they are stored as lists/Series of lists
    if not frame_df.empty and isinstance(frame_df[embedding_col].iloc[0], (list, pd.Series, np.ndarray)):
        try:
            # Attempt to stack, works well if elements are already ndarrays or lists of consistent length
            embeddings = np.stack(frame_df[embedding_col].values).astype(np.float32)
        except ValueError: # Fallback if elements are lists of varying types/lengths, or other complex structures
            try:
                embeddings = np.array(frame_df[embedding_col].tolist(), dtype=np.float32)
            except ValueError as e:
                 raise ValueError(f"Could not convert embedding column '{embedding_col}' to a 2D NumPy array. Ensure embeddings are uniform. Error: {e}")
        
        if embeddings.ndim == 1 and len(frame_df) > 1 : # Check if it became a 1D array of objects
            try: # This might happen if tolist() results in list of arrays that numpy doesn't stack directly into 2D
                embeddings = np.vstack(frame_df[embedding_col].values).astype(np.float32)
            except ValueError as e:
                 raise ValueError(f"Could not convert embedding column '{embedding_col}' to a 2D NumPy array even with vstack. Ensure embeddings are uniform. Error: {e}")


    else:
        raise ValueError(f"Embedding column '{embedding_col}' does not contain list-like or ndarray embeddings, or is empty.")

    if embeddings.shape[0] != len(frame_df):
        raise ValueError(f"Mismatch between number of embeddings ({embeddings.shape[0]}) and number of rows in DataFrame ({len(frame_df)}).")


    is_duplicate_flags = [False] * len(frame_df)
    if len(frame_df) < 2:
        frame_df['is_intra_video_duplicate'] = is_duplicate_flags
        return frame_df

    ref_embedding = embeddings[0]
    dissimilar_streak_count = 0

    for i in range(1, len(embeddings)):
        current_embedding = embeddings[i]
        sim = _calculate_cosine_similarity(ref_embedding, current_embedding)

        if sim >= similarity_threshold:
            is_duplicate_flags[i] = True
            dissimilar_streak_count = 0 # Reset streak as we found a duplicate of ref_embedding
        else:
            is_duplicate_flags[i] = False
            dissimilar_streak_count += 1
            # If enough dissimilar frames have passed, update the reference frame
            if dissimilar_streak_count >= min_sequence_break:
                ref_embedding = current_embedding
                dissimilar_streak_count = 0 # Reset for the new reference 
    
    frame_df['is_intra_video_duplicate'] = is_duplicate_flags
    return frame_df

# Example Usage (conceptual, actual use within the main Dask pipeline)
# if __name__ == '__main__':
#     # Create dummy data
#     num_frames = 10
#     embed_dim = 5 # Small embedding dim for example
#     data = []
#     # Frame 0, 1, 2 are similar
#     # Frame 3 is different
#     # Frame 4, 5 are similar to 3
#     # Frame 6, 7 different
#     # Frame 8, 9 similar to 7
#     base_embeds = [np.random.rand(embed_dim).astype(np.float32) for _ in range(5)]
#     base_embeds[1] = base_embeds[0] + 0.01 # Similar to 0
#     base_embeds[2] = base_embeds[0] - 0.01 # Similar to 0
#     base_embeds[4] = base_embeds[3] + 0.01 # Similar to 3
#     # Normalize them for cosine sim via dot product
#     base_embeds = [b / np.linalg.norm(b) if np.linalg.norm(b) > 0 else b for b in base_embeds]

#     embeddings_list = [
#         base_embeds[0], base_embeds[1], base_embeds[2], # 0,1,2
#         base_embeds[3], # 3
#         base_embeds[4], base_embeds[3], # 4,5 (5 is similar to 3, 4 is similar to 3)
#         np.random.rand(embed_dim).astype(np.float32), # 6 (different)
#         base_embeds[0] + 0.2, # 7 (make it somewhat different from 0, but could be new ref)
#         (base_embeds[0] + 0.2) + 0.01, # 8 (similar to 7)
#         (base_embeds[0] + 0.2) - 0.01  # 9 (similar to 7)
#     ]
#     embeddings_list = [b / np.linalg.norm(b) if np.linalg.norm(b) > 0 else b for b in embeddings_list]


#     mock_frame_df = pd.DataFrame({
#         'frame_id': [f'f{i:03d}' for i in range(num_frames)],
#         'image_embedding': embeddings_list
#     })
#     print("Original DataFrame:")
#     # for i, row in mock_frame_df.iterrows(): print(row['frame_id'], row['image_embedding'][:2])


#     deduped_df = deduplicate_video_frames_sequential(mock_frame_df.copy(), 
#                                                     embedding_col='image_embedding', 
#                                                     similarity_threshold=0.95,
#                                                     min_sequence_break=1)
#     print("\nSequential Deduplication (threshold=0.95, break=1):")
#     # Expected: f000 F, f001 T, f002 T, f003 F, f004 T, f005 T, f006 F, f007 F, f008 T, f009 T
#     for i, row_val in deduped_df.iterrows(): print(row_val['frame_id'], row_val['is_intra_video_duplicate'])
    
#     deduped_df_break2 = deduplicate_video_frames_sequential(mock_frame_df.copy(), 
#                                                     embedding_col='image_embedding', 
#                                                     similarity_threshold=0.95,
#                                                     min_sequence_break=2)
#     print("\nSequential Deduplication (threshold=0.95, break=2):")
#     # With break=2, frame 6 might still be compared to frame 3 if 3,4,5 were a sequence and 6 is different from 3.
#     # The logic keeps comparing to the 'ref_embedding' until dissimilar_streak_count >= min_sequence_break
#     # So if frame_3 is ref, frame_4 is dup, frame_5 is dup. frame_6 is different (streak=1 vs f3).
#     # frame_7 is different from frame_3 (streak=2 vs f3). New ref is frame_7.
#     # Expected: f000 F, f001 T, f002 T, f003 F, f004 T, f005 T, f006 F (streak=1 vs f3), f007 F (streak=2 vs f3, new ref=f7), f008 T, f009 T
#     for i, row_val in deduped_df_break2.iterrows(): print(row_val['frame_id'], row_val['is_intra_video_duplicate'])
