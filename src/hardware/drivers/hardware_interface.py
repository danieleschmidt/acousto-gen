"""
Hardware interface and drivers for transducer arrays.
Provides unified API for controlling various hardware platforms.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import serial
import socket
import struct
import time
import threading
from queue import Queue
import logging


logger = logging.getLogger(__name__)


@dataclass
class HardwareStatus:
    """Status information from hardware device."""
    connected: bool
    temperature: Optional[float] = None
    voltage: Optional[float] = None
    current: Optional[float] = None
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    uptime: Optional[float] = None
    
    def is_healthy(self) -> bool:
        """Check if hardware is in healthy state."""
        return self.connected and self.error_code is None


class HardwareInterface(ABC):
    """Abstract base class for hardware interfaces."""
    
    def __init__(self, name: str = "Generic Hardware"):
        """
        Initialize hardware interface.
        
        Args:
            name: Hardware identifier
        """
        self.name = name
        self.connected = False
        self.num_elements = 0
        self.max_frequency = 40e3
        self.min_frequency = 20e3
        self.current_phases = None
        self.current_amplitudes = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Command queue for asynchronous operation
        self._command_queue = Queue()
        self._worker_thread = None
        self._stop_worker = False
    
    @abstractmethod
    def connect(self, **kwargs) -> bool:
        """Connect to hardware device."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from hardware device."""
        pass
    
    @abstractmethod
    def send_phases(self, phases: np.ndarray) -> bool:
        """Send phase values to hardware."""
        pass
    
    @abstractmethod
    def send_amplitudes(self, amplitudes: np.ndarray) -> bool:
        """Send amplitude values to hardware."""
        pass
    
    @abstractmethod
    def get_status(self) -> HardwareStatus:
        """Get current hardware status."""
        pass
    
    def is_connected(self) -> bool:
        """Check if hardware is connected."""
        return self.connected
    
    def activate(self) -> bool:
        """Activate transducer output."""
        return self.send_control_command("ACTIVATE")
    
    def deactivate(self) -> bool:
        """Deactivate transducer output."""
        return self.send_control_command("DEACTIVATE")
    
    def emergency_stop(self):
        """Emergency stop - immediately disable all outputs."""
        logger.warning(f"Emergency stop triggered for {self.name}")
        self.deactivate()
        self.disconnect()
    
    def send_control_command(self, command: str, params: Optional[Dict] = None) -> bool:
        """
        Send control command to hardware.
        
        Args:
            command: Command string
            params: Optional command parameters
            
        Returns:
            Success status
        """
        logger.info(f"Sending command {command} to {self.name}")
        return True
    
    def start_worker(self):
        """Start background worker thread for command processing."""
        if self._worker_thread is None:
            self._stop_worker = False
            self._worker_thread = threading.Thread(target=self._worker_loop)
            self._worker_thread.daemon = True
            self._worker_thread.start()
    
    def stop_worker(self):
        """Stop background worker thread."""
        self._stop_worker = True
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None
    
    def _worker_loop(self):
        """Background worker loop for processing commands."""
        while not self._stop_worker:
            try:
                if not self._command_queue.empty():
                    command = self._command_queue.get(timeout=0.1)
                    self._process_command(command)
                else:
                    time.sleep(0.001)
            except Exception as e:
                logger.error(f"Worker thread error: {e}")
    
    def _process_command(self, command: Dict[str, Any]):
        """Process a command from the queue."""
        cmd_type = command.get('type')
        
        if cmd_type == 'phases':
            self.send_phases(command['data'])
        elif cmd_type == 'amplitudes':
            self.send_amplitudes(command['data'])
        elif cmd_type == 'control':
            self.send_control_command(command['command'], command.get('params'))


class SerialHardware(HardwareInterface):
    """Hardware interface using serial communication."""
    
    def __init__(
        self,
        name: str = "Serial Device",
        port: Optional[str] = None,
        baudrate: int = 115200
    ):
        """
        Initialize serial hardware interface.
        
        Args:
            name: Device name
            port: Serial port (e.g., '/dev/ttyUSB0' or 'COM3')
            baudrate: Serial baudrate
        """
        super().__init__(name)
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        
        # Protocol configuration
        self.protocol_version = 1
        self.packet_size = 1024
    
    def connect(self, port: Optional[str] = None, **kwargs) -> bool:
        """
        Connect to serial device.
        
        Args:
            port: Override port if specified
            
        Returns:
            Connection success status
        """
        if port:
            self.port = port
        
        if not self.port:
            logger.error("No serial port specified")
            return False
        
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1.0,
                write_timeout=1.0
            )
            
            # Send initialization sequence
            self._send_init_sequence()
            
            # Read device info
            self._read_device_info()
            
            self.connected = True
            logger.info(f"Connected to {self.name} on {self.port}")
            
            # Start worker thread
            self.start_worker()
            
            return True
            
        except serial.SerialException as e:
            logger.error(f"Failed to connect to {self.port}: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from serial device."""
        if self.serial_conn:
            # Stop worker thread
            self.stop_worker()
            
            # Send shutdown sequence
            self._send_shutdown_sequence()
            
            self.serial_conn.close()
            self.serial_conn = None
            self.connected = False
            logger.info(f"Disconnected from {self.name}")
    
    def send_phases(self, phases: np.ndarray) -> bool:
        """
        Send phase values over serial.
        
        Args:
            phases: Array of phase values in radians
            
        Returns:
            Success status
        """
        if not self.connected:
            return False
        
        with self._lock:
            # Convert to protocol format (e.g., 16-bit integers)
            phase_data = (phases * 65535 / (2 * np.pi)).astype(np.uint16)
            
            # Create packet
            packet = self._create_packet('PHASE', phase_data.tobytes())
            
            # Send packet
            try:
                self.serial_conn.write(packet)
                
                # Wait for acknowledgment
                ack = self.serial_conn.read(4)
                if ack == b'ACK\n':
                    self.current_phases = phases
                    return True
                else:
                    logger.warning(f"No ACK received for phase data")
                    return False
                    
            except serial.SerialException as e:
                logger.error(f"Failed to send phases: {e}")
                return False
    
    def send_amplitudes(self, amplitudes: np.ndarray) -> bool:
        """
        Send amplitude values over serial.
        
        Args:
            amplitudes: Array of amplitude values (0-1)
            
        Returns:
            Success status
        """
        if not self.connected:
            return False
        
        with self._lock:
            # Convert to protocol format (e.g., 8-bit integers)
            amp_data = (amplitudes * 255).astype(np.uint8)
            
            # Create packet
            packet = self._create_packet('AMPL', amp_data.tobytes())
            
            # Send packet
            try:
                self.serial_conn.write(packet)
                
                # Wait for acknowledgment
                ack = self.serial_conn.read(4)
                if ack == b'ACK\n':
                    self.current_amplitudes = amplitudes
                    return True
                else:
                    return False
                    
            except serial.SerialException as e:
                logger.error(f"Failed to send amplitudes: {e}")
                return False
    
    def get_status(self) -> HardwareStatus:
        """Get hardware status over serial."""
        if not self.connected:
            return HardwareStatus(connected=False)
        
        with self._lock:
            try:
                # Request status
                self.serial_conn.write(b'STATUS\n')
                
                # Read response
                response = self.serial_conn.readline()
                
                # Parse status
                status_data = self._parse_status(response)
                
                return HardwareStatus(
                    connected=True,
                    temperature=status_data.get('temp'),
                    voltage=status_data.get('voltage'),
                    current=status_data.get('current'),
                    error_code=status_data.get('error'),
                    uptime=status_data.get('uptime')
                )
                
            except serial.SerialException:
                return HardwareStatus(connected=False)
    
    def _create_packet(self, command: str, data: bytes) -> bytes:
        """Create protocol packet."""
        header = struct.pack('<4sHH', command.encode()[:4], len(data), 0)  # command, length, checksum
        checksum = sum(data) & 0xFFFF
        header = struct.pack('<4sHH', command.encode()[:4], len(data), checksum)
        return header + data
    
    def _parse_status(self, response: bytes) -> Dict[str, Any]:
        """Parse status response."""
        # Simple key-value parsing
        status = {}
        try:
            text = response.decode('utf-8').strip()
            for item in text.split(','):
                if '=' in item:
                    key, value = item.split('=')
                    try:
                        status[key] = float(value)
                    except ValueError:
                        status[key] = value
        except:
            pass
        return status
    
    def _send_init_sequence(self):
        """Send initialization sequence to device."""
        if self.serial_conn:
            self.serial_conn.write(b'INIT\n')
            time.sleep(0.1)
    
    def _send_shutdown_sequence(self):
        """Send shutdown sequence to device."""
        if self.serial_conn:
            self.serial_conn.write(b'SHUTDOWN\n')
            time.sleep(0.1)
    
    def _read_device_info(self):
        """Read device information."""
        if self.serial_conn:
            self.serial_conn.write(b'INFO\n')
            response = self.serial_conn.readline()
            # Parse device info
            info = self._parse_status(response)
            self.num_elements = int(info.get('elements', 256))


class NetworkHardware(HardwareInterface):
    """Hardware interface using network communication."""
    
    def __init__(
        self,
        name: str = "Network Device",
        host: Optional[str] = None,
        port: int = 8080
    ):
        """
        Initialize network hardware interface.
        
        Args:
            name: Device name
            host: Host address
            port: Network port
        """
        super().__init__(name)
        self.host = host
        self.port = port
        self.socket = None
        
        # Protocol configuration
        self.protocol = 'TCP'  # or 'UDP'
        self.buffer_size = 4096
    
    def connect(self, host: Optional[str] = None, port: Optional[int] = None, **kwargs) -> bool:
        """
        Connect to network device.
        
        Args:
            host: Override host if specified
            port: Override port if specified
            
        Returns:
            Connection success status
        """
        if host:
            self.host = host
        if port:
            self.port = port
        
        if not self.host:
            logger.error("No host address specified")
            return False
        
        try:
            if self.protocol == 'TCP':
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(5.0)
                self.socket.connect((self.host, self.port))
            else:  # UDP
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # Handshake
            self._handshake()
            
            self.connected = True
            logger.info(f"Connected to {self.name} at {self.host}:{self.port}")
            
            # Start worker thread
            self.start_worker()
            
            return True
            
        except socket.error as e:
            logger.error(f"Failed to connect to {self.host}:{self.port}: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from network device."""
        if self.socket:
            # Stop worker thread
            self.stop_worker()
            
            # Send disconnect message
            self._send_disconnect()
            
            self.socket.close()
            self.socket = None
            self.connected = False
            logger.info(f"Disconnected from {self.name}")
    
    def send_phases(self, phases: np.ndarray) -> bool:
        """Send phase values over network."""
        if not self.connected:
            return False
        
        with self._lock:
            # Convert to network format
            phase_data = (phases * 65535 / (2 * np.pi)).astype(np.uint16)
            
            # Create message
            message = self._create_message('PHASE', phase_data.tobytes())
            
            try:
                if self.protocol == 'TCP':
                    self.socket.send(message)
                    # Wait for ACK
                    response = self.socket.recv(16)
                    if response[:3] == b'ACK':
                        self.current_phases = phases
                        return True
                else:  # UDP
                    self.socket.sendto(message, (self.host, self.port))
                    self.current_phases = phases
                    return True
                    
            except socket.error as e:
                logger.error(f"Failed to send phases: {e}")
                return False
        
        return False
    
    def send_amplitudes(self, amplitudes: np.ndarray) -> bool:
        """Send amplitude values over network."""
        if not self.connected:
            return False
        
        with self._lock:
            # Convert to network format
            amp_data = (amplitudes * 255).astype(np.uint8)
            
            # Create message
            message = self._create_message('AMPL', amp_data.tobytes())
            
            try:
                if self.protocol == 'TCP':
                    self.socket.send(message)
                    # Wait for ACK
                    response = self.socket.recv(16)
                    if response[:3] == b'ACK':
                        self.current_amplitudes = amplitudes
                        return True
                else:  # UDP
                    self.socket.sendto(message, (self.host, self.port))
                    self.current_amplitudes = amplitudes
                    return True
                    
            except socket.error as e:
                logger.error(f"Failed to send amplitudes: {e}")
                return False
        
        return False
    
    def get_status(self) -> HardwareStatus:
        """Get hardware status over network."""
        if not self.connected:
            return HardwareStatus(connected=False)
        
        with self._lock:
            try:
                # Request status
                status_request = self._create_message('STATUS', b'')
                
                if self.protocol == 'TCP':
                    self.socket.send(status_request)
                    response = self.socket.recv(self.buffer_size)
                else:  # UDP
                    self.socket.sendto(status_request, (self.host, self.port))
                    response, _ = self.socket.recvfrom(self.buffer_size)
                
                # Parse status
                status_data = self._parse_network_status(response)
                
                return HardwareStatus(
                    connected=True,
                    temperature=status_data.get('temp'),
                    voltage=status_data.get('voltage'),
                    current=status_data.get('current'),
                    error_code=status_data.get('error')
                )
                
            except socket.error:
                return HardwareStatus(connected=False)
    
    def _create_message(self, command: str, data: bytes) -> bytes:
        """Create network protocol message."""
        # Simple protocol: [command(4)] [length(4)] [data]
        header = struct.pack('<4sI', command.encode()[:4], len(data))
        return header + data
    
    def _parse_network_status(self, response: bytes) -> Dict[str, Any]:
        """Parse network status response."""
        status = {}
        try:
            # Skip header
            data = response[8:]
            # Simple JSON parsing
            import json
            status = json.loads(data.decode('utf-8'))
        except:
            pass
        return status
    
    def _handshake(self):
        """Perform connection handshake."""
        if self.socket:
            handshake = self._create_message('HELLO', b'')
            if self.protocol == 'TCP':
                self.socket.send(handshake)
                response = self.socket.recv(16)
            else:
                self.socket.sendto(handshake, (self.host, self.port))
    
    def _send_disconnect(self):
        """Send disconnect message."""
        if self.socket:
            disconnect = self._create_message('BYE', b'')
            try:
                if self.protocol == 'TCP':
                    self.socket.send(disconnect)
                else:
                    self.socket.sendto(disconnect, (self.host, self.port))
            except:
                pass


class SimulationHardware(HardwareInterface):
    """Simulation hardware interface for testing and development."""
    
    def __init__(
        self,
        name: str = "Simulation",
        num_elements: int = 256,
        latency: float = 0.001
    ):
        """
        Initialize simulation hardware interface.
        
        Args:
            name: Device name
            num_elements: Number of transducer elements
            latency: Simulated communication latency
        """
        super().__init__(name)
        self.num_elements = num_elements
        self.latency = latency
        
        # Simulated hardware state
        self._simulated_temp = 23.5
        self._simulated_voltage = 12.0
        self._simulated_current = 0.5
        self._uptime = 0.0
        self._start_time = time.time()
        
        # Automatically connect in simulation
        self.connected = True
        logger.info(f"Initialized simulation hardware: {name} with {num_elements} elements")
    
    def connect(self, **kwargs) -> bool:
        """Connect to simulation (always succeeds)."""
        self.connected = True
        self._start_time = time.time()
        logger.info(f"Connected to simulation hardware: {self.name}")
        return True
    
    def disconnect(self):
        """Disconnect from simulation."""
        self.connected = False
        logger.info(f"Disconnected from simulation hardware: {self.name}")
    
    def send_phases(self, phases: np.ndarray) -> bool:
        """Simulate sending phase values."""
        if not self.connected:
            return False
        
        # Validate input
        if len(phases) != self.num_elements:
            logger.error(f"Expected {self.num_elements} phases, got {len(phases)}")
            return False
        
        # Simulate communication latency
        time.sleep(self.latency)
        
        # Store phases
        self.current_phases = phases.copy()
        
        # Simulate some thermal effects
        self._simulated_temp += 0.01 * np.mean(phases)
        self._simulated_current = 0.5 + 0.1 * np.std(phases)
        
        logger.debug(f"Sent {len(phases)} phase values to {self.name}")
        return True
    
    def send_amplitudes(self, amplitudes: np.ndarray) -> bool:
        """Simulate sending amplitude values."""
        if not self.connected:
            return False
        
        # Validate input
        if len(amplitudes) != self.num_elements:
            logger.error(f"Expected {self.num_elements} amplitudes, got {len(amplitudes)}")
            return False
        
        # Simulate communication latency
        time.sleep(self.latency)
        
        # Store amplitudes
        self.current_amplitudes = amplitudes.copy()
        
        # Simulate power consumption effects
        power = np.sum(amplitudes**2)
        self._simulated_current = 0.3 + power * 0.2
        self._simulated_temp += power * 0.01
        
        logger.debug(f"Sent {len(amplitudes)} amplitude values to {self.name}")
        return True
    
    def get_status(self) -> HardwareStatus:
        """Get simulated hardware status."""
        if not self.connected:
            return HardwareStatus(connected=False)
        
        # Update uptime
        self._uptime = time.time() - self._start_time
        
        # Simulate temperature cooling
        self._simulated_temp = max(20.0, self._simulated_temp - 0.01)
        
        # Simulate voltage fluctuations
        self._simulated_voltage = 12.0 + 0.1 * np.sin(time.time())
        
        return HardwareStatus(
            connected=True,
            temperature=self._simulated_temp,
            voltage=self._simulated_voltage,
            current=self._simulated_current,
            error_code=None,
            error_message=None,
            uptime=self._uptime
        )
    
    def activate(self) -> bool:
        """Simulate activation."""
        if not self.connected:
            return False
        
        logger.info(f"Activated {self.name}")
        return True
    
    def deactivate(self) -> bool:
        """Simulate deactivation."""
        if not self.connected:
            return False
        
        # Reset to safe state
        self._simulated_current = 0.1
        logger.info(f"Deactivated {self.name}")
        return True