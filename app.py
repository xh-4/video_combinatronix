#!/usr/bin/env python3
"""
AI Video Effects Web Interface
Clean, functional web interface for the AI-driven video processing system
"""

import os
import sys
import json
import base64
import io
import time
from typing import Dict, Any, List
from flask import Flask, render_template, request, jsonify, send_file, Response
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from intelligent_video_effects import IntelligentVideoProcessor, create_test_video
from Combinator_Kernel import load_video_chunks, VideoFrame, VideoChunk
from effects import create_temporal_blend_presets

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Global processor instance
processor = IntelligentVideoProcessor()

# Processing state
processing_state = {
    'is_processing': False,
    'current_video': None,
    'progress': 0,
    'results': [],
    'stats': {}
}

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = f"uploaded_{int(time.time())}.mp4"
        filepath = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)
        
        # Reset processing state
        processing_state['current_video'] = filepath
        processing_state['progress'] = 0
        processing_state['results'] = []
        processing_state['is_processing'] = False
        
        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'Video uploaded successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process', methods=['POST'])
def process_video():
    """Process video with AI effects"""
    try:
        data = request.get_json()
        video_path = data.get('video_path')
        chunk_size = data.get('chunk_size', 20)
        overlap = data.get('overlap', 5)
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 400
        
        # Start processing
        processing_state['is_processing'] = True
        processing_state['progress'] = 0
        processing_state['results'] = []
        
        # Process video chunks
        chunk_count = 0
        total_chunks = 0
        
        # Count total chunks first
        for _ in load_video_chunks(video_path, chunk_size, overlap):
            total_chunks += 1
        
        # Process chunks
        for chunk in load_video_chunks(video_path, chunk_size, overlap):
            chunk_count += 1
            
            # Process chunk with AI
            processed_frames, info = processor.process_chunk(chunk)
            
            # Store result
            result = {
                'chunk_id': chunk.chunk_id,
                'start_time': chunk.start_time,
                'end_time': chunk.end_time,
                'frame_count': len(processed_frames),
                'category': info['category'],
                'confidence': float(info['confidence']),
                'selected_effect': info['selected_effect'],
                'effect_description': info['effect_description'],
                'motion_level': float(info['context'].motion_level),
                'complexity_level': float(info['context'].complexity_level)
            }
            
            processing_state['results'].append(result)
            processing_state['progress'] = int((chunk_count / total_chunks) * 100)
            
            # Yield progress update
            yield f"data: {json.dumps({'progress': processing_state['progress'], 'chunk': chunk_count, 'total': total_chunks})}\n\n"
        
        # Train ESN on accumulated data
        processor.train_on_history()
        
        # Get final stats
        stats = processor.get_processing_stats()
        processing_state['stats'] = stats
        processing_state['is_processing'] = False
        
        yield f"data: {json.dumps({'complete': True, 'stats': stats})}\n\n"
        
    except Exception as e:
        processing_state['is_processing'] = False
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.route('/api/process_stream')
def process_stream():
    """Stream processing progress"""
    return Response(process_video(), mimetype='text/event-stream')

@app.route('/api/status')
def get_status():
    """Get current processing status"""
    return jsonify(processing_state)

@app.route('/api/results')
def get_results():
    """Get processing results"""
    return jsonify({
        'results': processing_state['results'],
        'stats': processing_state['stats']
    })

@app.route('/api/effects')
def get_effects():
    """Get available effects"""
    presets = create_temporal_blend_presets()
    effects = []
    
    for name, effect_func in presets.items():
        effects.append({
            'name': name,
            'description': effect_func.__name__,
            'type': 'temporal'
        })
    
    return jsonify({'effects': effects})

@app.route('/api/demo_video')
def create_demo_video():
    """Create a demo video for testing"""
    try:
        demo_path = "demo_video.mp4"
        create_test_video(demo_path, duration=3.0)
        
        return jsonify({
            'success': True,
            'video_path': demo_path,
            'message': 'Demo video created successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_chunk', methods=['POST'])
def process_single_chunk():
    """Process a single video chunk"""
    try:
        data = request.get_json()
        chunk_data = data.get('chunk')
        
        # Create VideoChunk from data
        frames = []
        for frame_data in chunk_data.get('frames', []):
            frame = VideoFrame(
                data=np.array(frame_data['data']),
                frame_number=frame_data['frame_number'],
                timestamp=frame_data['timestamp'],
                fps=frame_data['fps']
            )
            frames.append(frame)
        
        chunk = VideoChunk(
            frames=frames,
            chunk_id=chunk_data['chunk_id'],
            start_time=chunk_data['start_time'],
            end_time=chunk_data['end_time']
        )
        
        # Process chunk
        processed_frames, info = processor.process_chunk(chunk)
        
        # Convert processed frames to serializable format
        processed_data = []
        for field in processed_frames:
            processed_data.append({
                'real': field.I.tolist(),
                'imag': field.Q.tolist(),
                'magnitude': field.magnitude.tolist(),
                'phase': field.phase.tolist()
            })
        
        return jsonify({
            'success': True,
            'processed_frames': processed_data,
            'info': {
                'category': info['category'],
                'confidence': float(info['confidence']),
                'selected_effect': info['selected_effect'],
                'effect_description': info['effect_description']
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai_analysis', methods=['POST'])
def ai_analysis():
    """Perform AI analysis on video content"""
    try:
        data = request.get_json()
        video_path = data.get('video_path')
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 400
        
        # Analyze first few chunks
        analysis_results = []
        chunk_count = 0
        
        for chunk in load_video_chunks(video_path, chunk_size=15, overlap=3):
            if chunk_count >= 5:  # Limit analysis
                break
                
            processed_frames, info = processor.process_chunk(chunk)
            
            analysis_results.append({
                'chunk_id': chunk.chunk_id,
                'time_range': f"{chunk.start_time:.2f}s - {chunk.end_time:.2f}s",
                'category': info['category'],
                'confidence': float(info['confidence']),
                'motion_level': float(info['context'].motion_level),
                'complexity_level': float(info['context'].complexity_level),
                'recommended_effect': info['selected_effect'],
                'description': info['effect_description']
            })
            
            chunk_count += 1
        
        return jsonify({
            'success': True,
            'analysis': analysis_results,
            'summary': {
                'total_chunks_analyzed': len(analysis_results),
                'categories_found': list(set(r['category'] for r in analysis_results)),
                'effects_recommended': list(set(r['recommended_effect'] for r in analysis_results))
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("🚀 Starting AI Video Effects Web Interface...")
    print("📁 Upload directory created")
    print("🌐 Web interface ready at http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)



