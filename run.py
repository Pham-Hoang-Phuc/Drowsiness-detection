import subprocess
import os

# Define the path to drowsinessClient.exe
# Assuming the root is the current working directory
client_dir = os.path.join('DrowsinessClient', 'bin', 'Debug', 'net8.0-windows')
client_path = os.path.join(client_dir, 'drowsinessClient.exe')

# Check if the executable exists
if not os.path.exists(client_path):
    print(f"Error: {client_path} does not exist.")
    exit(1)

# Run drowsinessClient.exe in the background (non-blocking)
subprocess.Popen(client_path, cwd=client_dir)

# Run server.py (assuming it's in the root directory, and this is blocking)
subprocess.call(['python', 'server.py'])