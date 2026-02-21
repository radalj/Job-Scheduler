#!/usr/bin/env python3
"""
Concatenate two result files:
1. Read muxGNN_result_2.txt (keep original order)
2. Read muxGNN_result_end.txt and reverse line order
3. Append reversed content to the first file content
4. Write combined result to a new file
"""

def concat_files_with_reverse():
    file1 = "muxGNN_result_2.txt"
    file2 = "muxGNN_result_end.txt"
    output_file = "muxGNN_result_combined.txt"
    
    print(f"Reading {file1}...")
    # Read first file content
    try:
        with open(file1, "r", encoding="utf-8") as f:
            lines1 = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Successfully read {len(lines1)} lines from {file1}")
    except FileNotFoundError:
        print(f"Error: {file1} not found")
        lines1 = []
    except Exception as e:
        print(f"Error reading {file1}: {e}")
        lines1 = []
    
    print(f"Reading {file2}...")
    # Read second file content and reverse line order
    try:
        with open(file2, "r", encoding="utf-8") as f:
            lines2 = [line.strip() for line in f.readlines() if line.strip()]
        # Reverse the lines
        reversed_lines2 = list(reversed(lines2))
        print(f"Successfully read {len(lines2)} lines from {file2}, reversed them")
    except FileNotFoundError:
        print(f"Error: {file2} not found")
        reversed_lines2 = []
    except Exception as e:
        print(f"Error reading {file2}: {e}")
        reversed_lines2 = []
    
    print(f"Writing combined content to {output_file}...")
    # Combine content
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            # Write content from first file
            for line in lines1:
                f.write(line + '\n')
            
            # Write reversed content from second file
            for line in reversed_lines2:
                f.write(line + '\n')
        
        print(f"✓ Combined file created: {output_file}")
        print(f"  Lines from {file1}: {len(lines1)}")
        print(f"  Lines from {file2} (reversed): {len(reversed_lines2)}")
        print(f"  Total lines in output: {len(lines1) + len(reversed_lines2)}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

if __name__ == "__main__":
    concat_files_with_reverse()
