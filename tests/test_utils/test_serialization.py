"""
Tests for utils.serialization module
"""
import pytest
import numpy as np
import json
import tempfile
import os
from channelpy.core.state import State, StateArray, EMPTY, DELTA, PHI, PSI
from channelpy.core.nested import NestedState
from channelpy.core.parallel import ParallelChannels
from channelpy.utils.serialization import (
    ChannelPyEncoder, channelpy_object_hook, save_state, load_state,
    save_state_array, load_state_array, save_pipeline, load_pipeline,
    save_nested_state, load_nested_state, save_parallel_channels, load_parallel_channels,
    to_dict, from_dict, save, load
)


def test_channelpy_encoder():
    """Test ChannelPyEncoder"""
    encoder = ChannelPyEncoder()
    
    # Test State encoding
    state = State(1, 1)
    encoded = encoder.default(state)
    assert encoded['_type'] == 'State'
    assert encoded['i'] == 1
    assert encoded['q'] == 1
    
    # Test StateArray encoding
    state_array = StateArray.from_bits(i=[1, 0], q=[1, 1])
    encoded = encoder.default(state_array)
    assert encoded['_type'] == 'StateArray'
    assert encoded['i'] == [1, 0]
    assert encoded['q'] == [1, 1]


def test_channelpy_object_hook():
    """Test channelpy_object_hook"""
    # Test State decoding
    state_dict = {'_type': 'State', 'i': 1, 'q': 1}
    decoded = channelpy_object_hook(state_dict)
    assert isinstance(decoded, State)
    assert decoded.i == 1
    assert decoded.q == 1
    
    # Test StateArray decoding
    state_array_dict = {
        '_type': 'StateArray',
        'i': [1, 0],
        'q': [1, 1]
    }
    decoded = channelpy_object_hook(state_array_dict)
    assert isinstance(decoded, StateArray)
    assert len(decoded) == 2
    assert decoded[0] == PSI
    assert decoded[1] == PHI


def test_save_load_state():
    """Test save_state and load_state"""
    state = State(1, 0)
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        json_path = f.name
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pkl_path = f.name
    
    try:
        # Test JSON serialization
        save_state(state, json_path)
        loaded_state = load_state(json_path)
        assert loaded_state == state
        
        # Test pickle serialization
        save_state(state, pkl_path)
        loaded_state = load_state(pkl_path)
        assert loaded_state == state
        
    finally:
        os.unlink(json_path)
        os.unlink(pkl_path)


def test_save_load_state_array():
    """Test save_state_array and load_state_array"""
    state_array = StateArray.from_bits(i=[1, 0, 1], q=[1, 1, 0])
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        json_path = f.name
    
    try:
        # Test JSON serialization
        save_state_array(state_array, json_path)
        loaded_array = load_state_array(json_path)
        assert len(loaded_array) == len(state_array)
        assert all(loaded_array[i] == state_array[i] for i in range(len(state_array)))
        
    finally:
        os.unlink(json_path)


def test_save_load_nested_state():
    """Test save_nested_state and load_nested_state"""
    nested_state = NestedState(level0=PSI, level1=DELTA, level2=PHI)
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        json_path = f.name
    
    try:
        # Test JSON serialization
        save_nested_state(nested_state, json_path)
        loaded_state = load_nested_state(json_path)
        assert loaded_state.depth == nested_state.depth
        assert loaded_state.path_string() == nested_state.path_string()
        
    finally:
        os.unlink(json_path)


def test_save_load_parallel_channels():
    """Test save_parallel_channels and load_parallel_channels"""
    channels = ParallelChannels(
        technical=PSI,
        business=DELTA,
        team=PHI
    )
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        json_path = f.name
    
    try:
        # Test JSON serialization
        save_parallel_channels(channels, json_path)
        loaded_channels = load_parallel_channels(json_path)
        assert len(loaded_channels) == len(channels)
        assert loaded_channels['technical'] == channels['technical']
        assert loaded_channels['business'] == channels['business']
        assert loaded_channels['team'] == channels['team']
        
    finally:
        os.unlink(json_path)


def test_to_dict_from_dict():
    """Test to_dict and from_dict"""
    # Test State
    state = State(1, 1)
    state_dict = to_dict(state)
    assert state_dict['type'] == 'State'
    assert state_dict['i'] == 1
    assert state_dict['q'] == 1
    assert state_dict['symbol'] == 'ψ'
    
    reconstructed = from_dict(state_dict)
    assert reconstructed == state
    
    # Test StateArray
    state_array = StateArray.from_bits(i=[1, 0], q=[1, 1])
    array_dict = to_dict(state_array)
    assert array_dict['type'] == 'StateArray'
    assert array_dict['length'] == 2
    
    reconstructed = from_dict(array_dict)
    assert len(reconstructed) == len(state_array)
    assert all(reconstructed[i] == state_array[i] for i in range(len(state_array)))


def test_save_load_generic():
    """Test generic save and load functions"""
    state = State(1, 0)
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        json_path = f.name
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pkl_path = f.name
    
    try:
        # Test JSON
        save(state, json_path, format='json')
        loaded = load(json_path)
        assert loaded == state
        
        # Test pickle
        save(state, pkl_path, format='pickle')
        loaded = load(pkl_path)
        assert loaded == state
        
    finally:
        os.unlink(json_path)
        os.unlink(pkl_path)


def test_json_roundtrip():
    """Test JSON roundtrip for all object types"""
    objects = [
        State(1, 1),
        StateArray.from_bits(i=[1, 0], q=[1, 1]),
        NestedState(level0=PSI, level1=DELTA),
        ParallelChannels(tech=PSI, biz=DELTA)
    ]
    
    for obj in objects:
        # Convert to JSON and back
        json_str = json.dumps(obj, cls=ChannelPyEncoder)
        reconstructed = json.loads(json_str, object_hook=channelpy_object_hook)
        
        # Check basic properties
        if hasattr(obj, 'i') and hasattr(obj, 'q'):
            assert reconstructed.i == obj.i
            assert reconstructed.q == obj.q
        elif hasattr(obj, '__len__'):
            assert len(reconstructed) == len(obj)


def test_serialization_edge_cases():
    """Test serialization edge cases"""
    # Empty StateArray
    empty_array = StateArray.from_bits(i=[], q=[])
    assert len(empty_array) == 0
    
    # Single element arrays
    single_array = StateArray.from_bits(i=[1], q=[0])
    assert len(single_array) == 1
    assert single_array[0] == DELTA
    
    # Large arrays
    large_array = StateArray.from_bits(
        i=np.random.randint(0, 2, 1000),
        q=np.random.randint(0, 2, 1000)
    )
    assert len(large_array) == 1000







