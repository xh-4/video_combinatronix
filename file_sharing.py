#!/usr/bin/env python3
"""
Simple File Sharing Program
Allows drag and drop file sharing between two computers on a home network.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import socket
import threading
import os
import json
import time
from pathlib import Path
import shutil

class FileSharingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple File Sharing")
        self.root.geometry("600x500")
        
        # Network settings
        self.port = 12345
        self.host = self.get_local_ip()
        self.connected = False
        self.server_socket = None
        self.client_socket = None
        
        # File sharing directory
        self.share_dir = Path.cwd() / "shared_files"
        self.share_dir.mkdir(exist_ok=True)
        
        self.setup_ui()
        self.start_server()
        
    def get_local_ip(self):
        """Get the local IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Simple File Sharing", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Connection info
        info_frame = ttk.LabelFrame(main_frame, text="Connection Info", padding="10")
        info_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(info_frame, text=f"Your IP: {self.host}").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(info_frame, text=f"Port: {self.port}").grid(row=1, column=0, sticky=tk.W)
        
        # Connection controls
        conn_frame = ttk.LabelFrame(main_frame, text="Connect to Another Computer", padding="10")
        conn_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(conn_frame, text="Remote IP:").grid(row=0, column=0, sticky=tk.W)
        self.remote_ip_entry = ttk.Entry(conn_frame, width=20)
        self.remote_ip_entry.grid(row=0, column=1, padx=(5, 10))
        
        self.connect_btn = ttk.Button(conn_frame, text="Connect", command=self.connect_to_remote)
        self.connect_btn.grid(row=0, column=2)
        
        self.status_label = ttk.Label(conn_frame, text="Status: Server running, waiting for connection...")
        self.status_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # File operations
        file_frame = ttk.LabelFrame(main_frame, text="File Operations", padding="10")
        file_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(file_frame, text="Send File", command=self.send_file).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(file_frame, text="Refresh Files", command=self.refresh_files).grid(row=0, column=1)
        
        # File list
        list_frame = ttk.LabelFrame(main_frame, text="Shared Files", padding="10")
        list_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Create treeview for file list
        columns = ("Name", "Size", "Modified")
        self.file_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=10)
        
        for col in columns:
            self.file_tree.heading(col, text=col)
            self.file_tree.column(col, width=150)
        
        # Scrollbar for file list
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=scrollbar.set)
        
        self.file_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Drag and drop area
        drop_frame = ttk.LabelFrame(main_frame, text="Drag & Drop Files Here", padding="10")
        drop_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.drop_label = ttk.Label(drop_frame, text="Drop files here to share them", 
                                   relief="sunken", anchor="center")
        self.drop_label.grid(row=0, column=0, sticky=(tk.W, tk.E), ipady=20)
        
        # Configure drag and drop
        self.drop_label.bind("<Button-1>", self.on_drop_click)
        self.drop_label.bind("<B1-Motion>", self.on_drag_motion)
        self.drop_label.bind("<ButtonRelease-1>", self.on_drop_release)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        drop_frame.columnconfigure(0, weight=1)
        
        # Load initial files
        self.refresh_files()
    
    def start_server(self):
        """Start the file sharing server"""
        def server_thread():
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind((self.host, self.port))
                self.server_socket.listen(1)
                
                self.root.after(0, lambda: self.status_label.config(text="Status: Server running, waiting for connection..."))
                
                while True:
                    client_socket, addr = self.server_socket.accept()
                    self.client_socket = client_socket
                    self.connected = True
                    
                    self.root.after(0, lambda: self.status_label.config(text=f"Status: Connected to {addr[0]}"))
                    self.handle_client(client_socket)
                    
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(text=f"Status: Server error - {str(e)}"))
        
        threading.Thread(target=server_thread, daemon=True).start()
    
    def connect_to_remote(self):
        """Connect to a remote computer"""
        remote_ip = self.remote_ip_entry.get().strip()
        if not remote_ip:
            messagebox.showerror("Error", "Please enter a remote IP address")
            return
        
        def connect_thread():
            try:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect((remote_ip, self.port))
                self.connected = True
                
                self.root.after(0, lambda: self.status_label.config(text=f"Status: Connected to {remote_ip}"))
                self.handle_client(self.client_socket)
                
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(text=f"Status: Connection failed - {str(e)}"))
        
        threading.Thread(target=connect_thread, daemon=True).start()
    
    def handle_client(self, client_socket):
        """Handle client connection"""
        try:
            while self.connected:
                data = client_socket.recv(1024).decode()
                if not data:
                    break
                
                message = json.loads(data)
                
                if message["type"] == "file_list_request":
                    self.send_file_list(client_socket)
                elif message["type"] == "file_request":
                    self.send_file(client_socket, message["filename"])
                elif message["type"] == "file_data":
                    self.receive_file(message)
                    
        except Exception as e:
            print(f"Client handling error: {e}")
        finally:
            self.connected = False
            client_socket.close()
            self.root.after(0, lambda: self.status_label.config(text="Status: Disconnected"))
    
    def send_file_list(self, client_socket):
        """Send list of files to client"""
        files = []
        for file_path in self.share_dir.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "name": file_path.name,
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                })
        
        message = {
            "type": "file_list",
            "files": files
        }
        
        client_socket.send(json.dumps(message).encode())
    
    def send_file(self, client_socket=None, filename=None):
        """Send a file to the connected client"""
        if not self.connected:
            messagebox.showerror("Error", "Not connected to any computer")
            return
        
        if not filename:
            filename = filedialog.askopenfilename(title="Select file to send")
            if not filename:
                return
            filename = os.path.basename(filename)
        
        file_path = self.share_dir / filename
        if not file_path.exists():
            messagebox.showerror("Error", f"File {filename} not found in shared directory")
            return
        
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            message = {
                "type": "file_data",
                "filename": filename,
                "data": file_data.hex()
            }
            
            if client_socket:
                client_socket.send(json.dumps(message).encode())
            else:
                self.client_socket.send(json.dumps(message).encode())
            
            self.root.after(0, lambda: self.status_label.config(text=f"Status: Sent {filename}"))
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send file: {str(e)}")
    
    def receive_file(self, message):
        """Receive a file from client"""
        try:
            filename = message["filename"]
            file_data = bytes.fromhex(message["data"])
            
            file_path = self.share_dir / filename
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            self.root.after(0, self.refresh_files)
            self.root.after(0, lambda: self.status_label.config(text=f"Status: Received {filename}"))
            
        except Exception as e:
            print(f"Error receiving file: {e}")
    
    def refresh_files(self):
        """Refresh the file list display"""
        # Clear existing items
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        
        # Add files from shared directory
        for file_path in self.share_dir.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                size = self.format_file_size(stat.st_size)
                modified = time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))
                
                self.file_tree.insert("", "end", values=(file_path.name, size, modified))
    
    def format_file_size(self, size):
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    def on_drop_click(self, event):
        """Handle click on drop area"""
        self.drag_start = event.x, event.y
    
    def on_drag_motion(self, event):
        """Handle drag motion"""
        pass
    
    def on_drop_release(self, event):
        """Handle file drop"""
        # This is a simplified drag and drop - in a real implementation,
        # you'd use tkinterdnd2 or similar library for proper drag and drop
        filename = filedialog.askopenfilename(title="Select file to share")
        if filename:
            self.copy_file_to_share(filename)
    
    def copy_file_to_share(self, filepath):
        """Copy a file to the shared directory"""
        try:
            filename = os.path.basename(filepath)
            dest_path = self.share_dir / filename
            
            # If file exists, add timestamp to avoid conflicts
            if dest_path.exists():
                name, ext = os.path.splitext(filename)
                timestamp = int(time.time())
                filename = f"{name}_{timestamp}{ext}"
                dest_path = self.share_dir / filename
            
            shutil.copy2(filepath, dest_path)
            self.refresh_files()
            self.status_label.config(text=f"Status: Added {filename} to shared files")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy file: {str(e)}")

def main():
    root = tk.Tk()
    app = FileSharingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

