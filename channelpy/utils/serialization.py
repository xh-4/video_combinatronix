"""
Serialization utilities for channel algebra objects

Save and load states, pipelines, and other channel algebra objects
"""
import pickle
import json
import numpy as np
from typing import Any, Dict, Union, Optional
from pathlib import Path
import warnings

from ..core.state import State, StateArray, EMPTY, DELTA, PHI, PSI
from ..core.nested import NestedState
from ..core.parallel import ParallelChannels


class ChannelPyEncoder(json.JSONEncoder):
    """Custom JSON encoder for ChannelPy objects"""
    
    def default(self, obj):
        if isinstance(obj, State):
            return {
                '_type': 'State',
                'i': obj.i,
                'q': obj.q
            }
        elif isinstance(obj, StateArray):
            return {
                '_type': 'StateArray',
                'i': obj.i.tolist(),
                'q': obj.q.tolist()
            }
        elif isinstance(obj, NestedState):
            return {
                '_type': 'NestedState',
                'levels': {
                    f'level{i}': {'i': state.i, 'q': state.q}
                    for i, state in obj._levels.items()
                }
            }
        elif isinstance(obj, ParallelChannels):
            return {
                '_type': 'ParallelChannels',
                'channels': {
                    name: {'i': state.i, 'q': state.q}
                    for name, state in obj._channels.items()
                }
            }
        elif isinstance(obj, np.ndarray):
            return {
                '_type': 'ndarray',
                'data': obj.tolist(),
                'dtype': str(obj.dtype)
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        
        return super().default(obj)


def channelpy_object_hook(dct):
    """Custom JSON decoder for ChannelPy objects"""
    if '_type' in dct:
        if dct['_type'] == 'State':
            return State(dct['i'], dct['q'])
        elif dct['_type'] == 'StateArray':
            return StateArray(
                np.array(dct['i'], dtype=np.int8),
                np.array(dct['q'], dtype=np.int8)
            )
        elif dct['_type'] == 'NestedState':
            levels = {}
            for level_key, state_data in dct['levels'].items():
                level_num = int(level_key[5:])  # Remove 'level' prefix
                levels[level_num] = State(state_data['i'], state_data['q'])
            return NestedState(**{f'level{i}': state for i, state in levels.items()})
        elif dct['_type'] == 'ParallelChannels':
            channels = {
                name: State(state_data['i'], state_data['q'])
                for name, state_data in dct['channels'].items()
            }
            return ParallelChannels(**channels)
        elif dct['_type'] == 'ndarray':
            return np.array(dct['data'], dtype=dct['dtype'])
    
    return dct


def save_state(state: State, filepath: Union[str, Path]) -> None:
    """
    Save a single State to file
    
    Parameters
    ----------
    state : State
        State to save
    filepath : str or Path
        Path to save file
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(state, f, cls=ChannelPyEncoder, indent=2)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)


def load_state(filepath: Union[str, Path]) -> State:
    """
    Load a single State from file
    
    Parameters
    ----------
    filepath : str or Path
        Path to load file from
        
    Returns
    -------
    State
        Loaded state
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f, object_hook=channelpy_object_hook)
    else:
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def save_state_array(state_array: StateArray, filepath: Union[str, Path]) -> None:
    """
    Save StateArray to file
    
    Parameters
    ----------
    state_array : StateArray
        StateArray to save
    filepath : str or Path
        Path to save file
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(state_array, f, cls=ChannelPyEncoder, indent=2)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(state_array, f)


def load_state_array(filepath: Union[str, Path]) -> StateArray:
    """
    Load StateArray from file
    
    Parameters
    ----------
    filepath : str or Path
        Path to load file from
        
    Returns
    -------
    StateArray
        Loaded state array
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f, object_hook=channelpy_object_hook)
    else:
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def save_pipeline(pipeline, filepath: Union[str, Path]) -> None:
    """
    Save pipeline to file
    
    Parameters
    ----------
    pipeline : BasePipeline
        Pipeline to save
    filepath : str or Path
        Path to save file
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        # JSON serialization for pipelines is limited
        warnings.warn(
            "JSON serialization of pipelines is limited. "
            "Use pickle for full pipeline serialization.",
            UserWarning
        )
        # Save basic info only
        pipeline_info = {
            'type': type(pipeline).__name__,
            'is_fitted': pipeline.is_fitted,
            'preprocessors': len(pipeline.preprocessors),
            'encoders': len(pipeline.encoders),
            'interpreters': len(pipeline.interpreters)
        }
        with open(filepath, 'w') as f:
            json.dump(pipeline_info, f, indent=2)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline, f)


def load_pipeline(filepath: Union[str, Path]):
    """
    Load pipeline from file
    
    Parameters
    ----------
    filepath : str or Path
        Path to load file from
        
    Returns
    -------
    BasePipeline
        Loaded pipeline
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        raise ValueError(
            "Cannot load full pipeline from JSON. "
            "Use pickle format for pipeline serialization."
        )
    else:
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def save_nested_state(nested_state: NestedState, filepath: Union[str, Path]) -> None:
    """
    Save NestedState to file
    
    Parameters
    ----------
    nested_state : NestedState
        NestedState to save
    filepath : str or Path
        Path to save file
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(nested_state, f, cls=ChannelPyEncoder, indent=2)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(nested_state, f)


def load_nested_state(filepath: Union[str, Path]) -> NestedState:
    """
    Load NestedState from file
    
    Parameters
    ----------
    filepath : str or Path
        Path to load file from
        
    Returns
    -------
    NestedState
        Loaded nested state
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f, object_hook=channelpy_object_hook)
    else:
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def save_parallel_channels(channels: ParallelChannels, filepath: Union[str, Path]) -> None:
    """
    Save ParallelChannels to file
    
    Parameters
    ----------
    channels : ParallelChannels
        ParallelChannels to save
    filepath : str or Path
        Path to save file
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(channels, f, cls=ChannelPyEncoder, indent=2)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(channels, f)


def load_parallel_channels(filepath: Union[str, Path]) -> ParallelChannels:
    """
    Load ParallelChannels from file
    
    Parameters
    ----------
    filepath : str or Path
        Path to load file from
        
    Returns
    -------
    ParallelChannels
        Loaded parallel channels
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f, object_hook=channelpy_object_hook)
    else:
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def to_dict(obj: Any) -> Dict:
    """
    Convert ChannelPy object to dictionary
    
    Parameters
    ----------
    obj : Any
        Object to convert
        
    Returns
    -------
    dict
        Dictionary representation
    """
    if isinstance(obj, State):
        return {
            'type': 'State',
            'i': obj.i,
            'q': obj.q,
            'symbol': str(obj)
        }
    elif isinstance(obj, StateArray):
        return {
            'type': 'StateArray',
            'i': obj.i.tolist(),
            'q': obj.q.tolist(),
            'length': len(obj),
            'symbols': obj.to_strings().tolist()
        }
    elif isinstance(obj, NestedState):
        return {
            'type': 'NestedState',
            'depth': obj.depth,
            'path': obj.path_string(),
            'levels': {
                f'level{i}': {'i': state.i, 'q': state.q, 'symbol': str(state)}
                for i, state in obj._levels.items()
            }
        }
    elif isinstance(obj, ParallelChannels):
        return {
            'type': 'ParallelChannels',
            'channels': {
                name: {'i': state.i, 'q': state.q, 'symbol': str(state)}
                for name, state in obj._channels.items()
            }
        }
    else:
        return {'type': type(obj).__name__, 'value': str(obj)}


def from_dict(dct: Dict) -> Any:
    """
    Convert dictionary to ChannelPy object
    
    Parameters
    ----------
    dct : dict
        Dictionary representation
        
    Returns
    -------
    Any
        Reconstructed object
    """
    obj_type = dct.get('type')
    
    if obj_type == 'State':
        return State(dct['i'], dct['q'])
    elif obj_type == 'StateArray':
        return StateArray(
            np.array(dct['i'], dtype=np.int8),
            np.array(dct['q'], dtype=np.int8)
        )
    elif obj_type == 'NestedState':
        levels = {}
        for level_key, state_data in dct['levels'].items():
            level_num = int(level_key[5:])  # Remove 'level' prefix
            levels[level_num] = State(state_data['i'], state_data['q'])
        return NestedState(**{f'level{i}': state for i, state in levels.items()})
    elif obj_type == 'ParallelChannels':
        channels = {
            name: State(state_data['i'], state_data['q'])
            for name, state_data in dct['channels'].items()
        }
        return ParallelChannels(**channels)
    else:
        return dct.get('value', dct)


# Convenience functions

def save(obj: Any, filepath: Union[str, Path], format: str = 'auto') -> None:
    """
    Save any ChannelPy object to file
    
    Parameters
    ----------
    obj : Any
        Object to save
    filepath : str or Path
        Path to save file
    format : str
        'auto', 'json', or 'pickle'
    """
    filepath = Path(filepath)
    
    if format == 'auto':
        format = 'json' if filepath.suffix == '.json' else 'pickle'
    
    if format == 'json':
        with open(filepath, 'w') as f:
            json.dump(obj, f, cls=ChannelPyEncoder, indent=2)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)


def load(filepath: Union[str, Path]) -> Any:
    """
    Load any ChannelPy object from file
    
    Parameters
    ----------
    filepath : str or Path
        Path to load file from
        
    Returns
    -------
    Any
        Loaded object
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f, object_hook=channelpy_object_hook)
    else:
        with open(filepath, 'rb') as f:
            return pickle.load(f)







