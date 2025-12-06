"""
Remote Policy Client for Evaluation

This module provides the RemotePolicyClient class that acts like a MetaPolicy 
but communicates with a remote policy server.
"""

import socket
import pickle
import struct
import re
import os
from collections import deque
from typing import Optional, List
import numpy as np
from loguru import logger
from benchmark.base import MetaObs, MetaAction


class PolicyClient:
    """
    Remote Policy Client that acts like a MetaPolicy but communicates with a policy server.
    
    This class mimics the interface of MetaPolicy but sends observations to a remote server
    and receives action chunks back. It maintains an action queue just like MetaPolicy.
    """
    
    def __init__(self, host: str, port: int, chunk_size: Optional[int] = None, ctrl_space: str = 'ee', ctrl_type: str = 'delta'):
        self.host = host
        self.port = port
        self.chunk_size = chunk_size
        self.ctrl_space = ctrl_space
        self.ctrl_type = ctrl_type
        if chunk_size is not None and chunk_size>0:
            self.action_queue = deque(maxlen=chunk_size)
        else:
            self.action_queue = deque(maxlen=1000)
        self.socket: Optional[socket.socket] = None
        
        # Connect to server
        self._connect()
        
        logger.info(f"✓ Connected to policy server at {host}:{port}")
        logger.info(f"  Chunk size: {chunk_size}")
    
    def _connect(self):
        """Connect to the policy server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
        except Exception as e:
            raise ConnectionError(f"Failed to connect to policy server at {self.host}:{self.port}: {e}")
    
    def _disconnect(self):
        """Disconnect from the policy server"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
    
    def _send_meta_obs(self, meta_obs: MetaObs) -> List[MetaAction]:
        """
        Send MetaObs to server and receive MetaAction list.
        
        Args:
            meta_obs: MetaObs object to send
            
        Returns:
            List of MetaAction objects
        """
        if not self.socket:
            raise RuntimeError("Not connected to server")
        
        try:
            # Serialize MetaObs
            data_bytes = pickle.dumps(meta_obs)
            data_length = len(data_bytes)
            
            # Send length (4 bytes, big-endian)
            length_bytes = struct.pack('>I', data_length)
            self.socket.sendall(length_bytes)
            
            # Send data
            self.socket.sendall(data_bytes)
            
            # Receive response
            return self._receive_mact_list()
            
        except Exception as e:
            logger.error(f"✗ Error communicating with server: {e}")
            # Try to reconnect once
            try:
                self._disconnect()
                self._connect()
                logger.info("✓ Reconnected to server, retrying...")
                return self._send_meta_obs(meta_obs)
            except:
                raise RuntimeError(f"Failed to communicate with server and reconnection failed: {e}")
    
    def _receive_mact_list(self) -> List[MetaAction]:
        """
        Receive list of MetaAction from server.
        
        Returns:
            List of MetaAction objects
        """
        try:
            # Read 4 bytes for data length
            length_bytes = self._recv_exactly(4)
            if not length_bytes:
                return []
            
            data_length = struct.unpack('>I', length_bytes)[0]
            
            # Read the actual data
            data_bytes = self._recv_exactly(data_length)
            if not data_bytes:
                return []
            
            # Deserialize
            mact_list = pickle.loads(data_bytes)
            return mact_list
            
        except Exception as e:
            logger.error(f"✗ Error receiving MetaAction list: {e}")
            return []
    
    def _recv_exactly(self, num_bytes: int) -> Optional[bytes]:
        """
        Receive exactly num_bytes from socket.
        
        Returns:
            bytes or None if connection closed
        """
        data = b''
        while len(data) < num_bytes:
            chunk = self.socket.recv(num_bytes - len(data))
            if not chunk:
                return None
            data += chunk
        return data
    
    def is_action_queue_empty(self):
        """Check if action queue is empty"""
        return len(self.action_queue) == 0
    
    def select_action(self, mobs: MetaObs, t: int, return_all=False):
        """
        Select action from remote policy server.
        
        This method mimics MetaPolicy.select_action():
        - When chunk is needed (t % chunk_size == 0 or queue empty), request new chunk from server
        - Otherwise, use actions from the queue
        - Block and wait for server response when queue is empty
        
        Args:
            mobs: MetaObs observation
            t: Current timestep
            return_all: Whether to return all remaining actions
            
        Returns:
            Action array or list of actions (compatible with evaluate function)
        """
        # Determine if we need to request new actions
        should_infer = len(self.action_queue) == 0  # Always request when queue is empty
        
        # If chunk_size is valid (>0), also check if it's time to request based on timestep
        if self.chunk_size is not None and self.chunk_size > 0:
            should_infer = should_infer or (t % self.chunk_size == 0)
        
        # Request new chunk when needed
        if should_infer:
            # Set timestep in observation (match MetaPolicy format)
            if hasattr(mobs, 'state') and mobs.state is not None:
                batch_size = mobs.state.shape[0] if len(mobs.state.shape) > 1 else 1
                mobs.timestep = np.array([[t] for _ in range(batch_size)])
            else:
                mobs.timestep = np.array([[t]])
            
            logger.debug(f"  📤 Requesting new action chunk from server (t={t})")
            
            # Send observation to server and get action chunk
            mact_list = self._send_meta_obs(mobs)
            
            if not mact_list:
                raise RuntimeError("Server returned empty action list")
            
            logger.debug(f"  📥 Received {len(mact_list)} actions from server")
            
            # Clear existing queue and add new actions
            while len(self.action_queue) > 0:
                self.action_queue.popleft()
            
            # Add actions to queue
            # If chunk_size > 0, limit to chunk_size; if chunk_size <= 0, use all actions
            actions_to_add = mact_list[:self.chunk_size] if self.chunk_size > 0 else mact_list
            for mact in actions_to_add:
                self.action_queue.append(mact)
        
        # Return actions from queue
        if return_all:
            all_macts = []
            while len(self.action_queue) > 0:
                all_macts.append(self.action_queue.popleft())
            return np.concatenate(all_macts) if all_macts else np.array([])
        
        # Return single action
        if len(self.action_queue) == 0:
            raise RuntimeError("Action queue is empty and server request failed")
        
        mact = self.action_queue.popleft()
        
        # The evaluate function expects the same format as MetaPolicy.select_action returns
        # MetaPolicy returns a numpy array of MetaAction objects
        return mact
    
    def reset(self):
        """Reset the policy (clear action queue)"""
        self.action_queue.clear()
        logger.debug("  🔄 Remote policy reset (cleared action queue)")
    
    def __del__(self):
        """Cleanup on destruction"""
        self._disconnect()


def parse_server_address(model_path: str) -> tuple:
    """
    Parse server address from model path.
    
    Expected format: "host:port" or "ip:port"
    
    Args:
        model_path: String in format "host:port"
        
    Returns:
        tuple: (host, port)
        
    Raises:
        ValueError: If format is invalid
    """
    # Check if it looks like a server address (contains colon and port number)
    if ':' in model_path and not os.path.exists(model_path):
        # Try to parse as host:port
        match = re.match(r'^(.+):(\d+)$', model_path)
        if match:
            host = match.group(1)
            port = int(match.group(2))
            return host, port
    
    raise ValueError(f"Invalid server address format: {model_path}. Expected format: 'host:port'")


def is_server_address(model_path: str) -> bool:
    """
    Check if model_path looks like a server address.
    
    Args:
        model_path: Path or server address string
        
    Returns:
        bool: True if it looks like a server address
    """
    try:
        parse_server_address(model_path)
        return True
    except ValueError:
        return False
