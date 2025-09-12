# Simple File Sharing Program

A lightweight file sharing application that allows you to share files between two computers on a home network with drag and drop functionality.

## Features

- **Easy Setup**: No external dependencies required
- **Drag & Drop**: Simple file sharing interface
- **Network Discovery**: Automatically finds your local IP
- **Real-time Transfer**: Files are shared instantly between connected computers
- **File Management**: View and manage shared files in a clean interface

## How to Use

### Step 1: Run the Program
```bash
python file_sharing.py
```

### Step 2: Connect Computers
1. **Computer A**: Run the program - it will start as a server
2. **Computer B**: Run the program and enter Computer A's IP address in the "Remote IP" field
3. Click "Connect" to establish connection

### Step 3: Share Files
- **Method 1**: Click "Send File" button to select and send files
- **Method 2**: Click the "Drag & Drop Files Here" area to select files
- **Method 3**: Copy files directly to the `shared_files` folder in the program directory

## File Management

- All shared files are stored in the `shared_files` folder
- The program automatically creates this folder if it doesn't exist
- Files are displayed with name, size, and modification date
- Click "Refresh Files" to update the file list

## Network Requirements

- Both computers must be on the same local network
- Default port: 12345 (automatically configured)
- No firewall configuration needed for most home networks

## Troubleshooting

- **Connection Failed**: Check that both computers are on the same network
- **Files Not Appearing**: Click "Refresh Files" to update the list
- **Permission Errors**: Ensure the program has write access to its directory

## Technical Details

- Uses Python's built-in `tkinter` for GUI
- Socket-based networking for file transfer
- JSON protocol for communication
- Threaded operations for non-blocking UI
- Cross-platform compatible (Windows, Mac, Linux)

