"""
Mock OpenCV implementation for testing when OpenCV is not available
"""

import numpy as np
from typing import Tuple, Any

class MockVideoCapture:
    """Mock VideoCapture class"""
    
    def __init__(self, filename: str = None):
        self.filename = filename
        self.frame_count = 0
        self.fps = 30.0
        self.width = 640
        self.height = 480
        self.total_frames = 60  # 2 seconds at 30fps
        
    def read(self) -> Tuple[bool, np.ndarray]:
        """Read next frame"""
        if self.frame_count >= self.total_frames:
            return False, None
        
        # Create synthetic frame
        frame = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        
        # Add some pattern
        center_x = int(self.width * (0.5 + 0.3 * np.sin(2 * np.pi * self.frame_count / 30)))
        center_y = int(self.height * (0.5 + 0.3 * np.cos(2 * np.pi * self.frame_count / 30)))
        
        # Draw circle
        y, x = np.ogrid[:self.height, :self.width]
        mask = (x - center_x)**2 + (y - center_y)**2 <= 50**2
        frame[mask] = [0, 255, 0]  # Green circle
        
        self.frame_count += 1
        return True, frame
    
    def get(self, prop: int) -> float:
        """Get property value"""
        if prop == 5:  # CAP_PROP_FPS
            return self.fps
        elif prop == 3:  # CAP_PROP_FRAME_WIDTH
            return self.width
        elif prop == 4:  # CAP_PROP_FRAME_HEIGHT
            return self.height
        return 0.0
    
    def release(self):
        """Release capture"""
        pass

class MockVideoWriter:
    """Mock VideoWriter class"""
    
    def __init__(self, filename: str, fourcc: int, fps: float, frame_size: Tuple[int, int]):
        self.filename = filename
        self.fourcc = fourcc
        self.fps = fps
        self.frame_size = frame_size
        self.frames_written = 0
        
    def write(self, frame: np.ndarray):
        """Write frame"""
        self.frames_written += 1
        
    def release(self):
        """Release writer"""
        print(f"Mock video written: {self.filename} ({self.frames_written} frames)")

# Mock constants
CAP_PROP_FPS = 5
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4

def VideoCapture(filename: str = None) -> MockVideoCapture:
    """Create mock VideoCapture"""
    return MockVideoCapture(filename)

def VideoWriter(filename: str, fourcc: int, fps: float, frame_size: Tuple[int, int]) -> MockVideoWriter:
    """Create mock VideoWriter"""
    return MockVideoWriter(filename, fourcc, fps, frame_size)

def VideoWriter_fourcc(*args) -> int:
    """Mock fourcc function"""
    return hash(''.join(args)) % 1000000

# Mock drawing functions
def circle(img: np.ndarray, center: Tuple[int, int], radius: int, color: Tuple[int, int, int], thickness: int = -1) -> np.ndarray:
    """Draw circle on image"""
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    img[mask] = color
    return img

def rectangle(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], thickness: int = -1) -> np.ndarray:
    """Draw rectangle on image"""
    img[pt1[1]:pt2[1], pt1[0]:pt2[0]] = color
    return img

def putText(img: np.ndarray, text: str, org: Tuple[int, int], font: int, fontScale: float, color: Tuple[int, int, int], thickness: int = 1) -> np.ndarray:
    """Put text on image (simplified)"""
    # Just add a simple text pattern
    x, y = org
    for i, char in enumerate(text[:20]):  # Limit text length
        if x + i * 10 < img.shape[1]:
            img[y:y+20, x+i*10:x+(i+1)*10] = color
    return img

# Mock font constants
FONT_HERSHEY_SIMPLEX = 0

# Mock version
__version__ = "4.8.0-mock"
