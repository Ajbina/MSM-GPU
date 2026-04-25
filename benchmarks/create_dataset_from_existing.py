#!/usr/bin/env python3
"""
Create N=35M dataset by combining existing datasets
Uses N=20M + N=15M worth of data from N=50M
"""
import struct
import os

def get_dataset_sizes():
    """Get sizes of scalar and G1J point in bytes"""
    # From main.cu: ScalarR has v[4] as uint64[4], and G1J is affine point
    scalar_size = 4 * 8  # 4 x uint64
    point_size = 6 * 8   # 6 x uint64 (for affine x, y, plus flags)
    return scalar_size, point_size

def create_35m_from_50m():
    """Load N=50M dataset and extract first 35M entries"""
    source_file = "/home/syn324/projects/msm-gpu/benchmarks/dataset_N50000000.bin"
    target_file = "/home/syn324/projects/msm-gpu/benchmarks/dataset_N35000000.bin"
    
    source_N = 50000000
    target_N = 35000000
    scalar_size, point_size = get_dataset_sizes()
    entry_size = scalar_size + point_size
    
    print(f"Creating N={target_N} dataset from N={source_N}...")
    print(f"Scalar size: {scalar_size} bytes, Point size: {point_size} bytes")
    print(f"Entry size: {entry_size} bytes")
    
    with open(source_file, 'rb') as src:
        # Read and verify source N
        src_N_bytes = src.read(4)
        src_N = struct.unpack('i', src_N_bytes)[0]
        print(f"Source file N: {src_N}")
        
        if src_N != source_N:
            print(f"ERROR: Source N mismatch! Expected {source_N}, got {src_N}")
            return False
        
        # Create target with target_N
        with open(target_file, 'wb') as tgt:
            tgt.write(struct.pack('i', target_N))
            
            # Copy target_N entries (each entry = scalar + point)
            bytes_to_copy = target_N * entry_size
            bytes_copied = 0
            chunk_size = 1024 * 1024 * 100  # 100MB chunks
            
            while bytes_copied < bytes_to_copy:
                to_read = min(chunk_size, bytes_to_copy - bytes_copied)
                chunk = src.read(to_read)
                if not chunk:
                    print(f"ERROR: Premature EOF at {bytes_copied} bytes")
                    return False
                tgt.write(chunk)
                bytes_copied += len(chunk)
                print(f"  Copied {bytes_copied / (1024*1024):.1f} MB / {bytes_to_copy / (1024*1024):.1f} MB")
    
    # Verify
    file_size = os.path.getsize(target_file)
    expected_size = 4 + (target_N * entry_size)  # 4-byte N header + entries
    print(f"\nDataset created: {target_file}")
    print(f"File size: {file_size / (1024*1024):.1f} MB")
    print(f"Expected size: {expected_size / (1024*1024):.1f} MB")
    
    if file_size == expected_size:
        print("✓ Dataset valid!")
        return True
    else:
        print(f"WARNING: Size mismatch! Expected {expected_size}, got {file_size}")
        return False

if __name__ == "__main__":
    create_35m_from_50m()
