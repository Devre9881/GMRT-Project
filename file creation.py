import numpy as np
import os

def create_test_file(filename, reference_file):
    size = os.path.getsize(reference_file)  # Get exact size of the reference file
    data = np.random.randint(-128, 127, size, dtype=np.int8)  # Generate random int8 values
    data.tofile(filename)
    print(f"File created: {filename} ({size} bytes)")

# Create a test file with the same size as pol1_2024_12_26_11_52_AM.raw
create_test_file("test_file.raw", "pol1_2024_12_26_11_52_AM.raw")
