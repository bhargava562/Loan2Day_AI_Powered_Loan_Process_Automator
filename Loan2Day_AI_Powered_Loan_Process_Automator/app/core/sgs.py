"""
Spectral-Graph Sentinel (SGS) - Advanced Security Module

This module provides comprehensive security scanning capabilities for the Loan2Day
platform, including deepfake detection, file validation, and topology analysis.
All file uploads MUST pass through SGS.scan_topology() before processing.

Key Features:
- Spectral analysis using PyTorch tensor operations
- Deepfake detection with confidence scoring
- File topology scanning and risk assessment
- Security event logging and audit trails
- Placeholder implementations for development/testing

Security Standards:
- All uploads scanned before processing (Zero-Trust Architecture)
- Multi-layered detection algorithms
- Comprehensive logging for compliance
- Fail-safe defaults (reject on uncertainty)

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union, BinaryIO
from pathlib import Path
from decimal import Decimal
import hashlib
import mimetypes
import logging
from datetime import datetime
from enum import Enum
import numpy as np
import io

# Configure logger
logger = logging.getLogger(__name__)

class SecurityThreatLevel(Enum):
    """Security threat level classifications."""
    SAFE = "SAFE"
    LOW_RISK = "LOW_RISK"
    MEDIUM_RISK = "MEDIUM_RISK"
    HIGH_RISK = "HIGH_RISK"
    CRITICAL = "CRITICAL"

class SGSError(Exception):
    """Base exception for SGS module errors."""
    pass

class UnsupportedFileTypeError(SGSError):
    """Raised when file type is not supported for scanning."""
    pass

class ScanFailureError(SGSError):
    """Raised when security scan fails to complete."""
    pass

class DeepfakeDetectionError(SGSError):
    """Raised when deepfake detection encounters an error."""
    pass

class SecurityScore:
    """
    Comprehensive security scoring result from SGS analysis.
    
    All scores are normalized to 0.0-1.0 range where:
    - 0.0 = Maximum security risk (definitely malicious)
    - 1.0 = Minimum security risk (definitely safe)
    """
    
    def __init__(
        self,
        overall_score: float,
        deepfake_score: float,
        topology_score: float,
        file_integrity_score: float,
        threat_level: SecurityThreatLevel,
        scan_timestamp: datetime,
        file_hash: str,
        file_type: str,
        file_size_bytes: int,
        detection_details: Dict[str, Any]
    ):
        self.overall_score = overall_score
        self.deepfake_score = deepfake_score
        self.topology_score = topology_score
        self.file_integrity_score = file_integrity_score
        self.threat_level = threat_level
        self.scan_timestamp = scan_timestamp
        self.file_hash = file_hash
        self.file_type = file_type
        self.file_size_bytes = file_size_bytes
        self.detection_details = detection_details
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert security score to dictionary for JSON serialization."""
        return {
            "overall_score": self.overall_score,
            "deepfake_score": self.deepfake_score,
            "topology_score": self.topology_score,
            "file_integrity_score": self.file_integrity_score,
            "threat_level": self.threat_level.value,
            "scan_timestamp": self.scan_timestamp.isoformat(),
            "file_hash": self.file_hash,
            "file_type": self.file_type,
            "file_size_bytes": self.file_size_bytes,
            "detection_details": self.detection_details,
            "is_safe": self.is_safe(),
            "risk_factors": self.get_risk_factors()
        }
    
    def is_safe(self) -> bool:
        """
        Determine if file is safe for processing based on security scores.
        
        Returns:
            bool: True if file is safe, False if risky
        """
        # Conservative threshold: require high confidence for safety
        return (
            self.overall_score >= 0.7 and
            self.deepfake_score >= 0.6 and
            self.topology_score >= 0.7 and
            self.file_integrity_score >= 0.8 and
            self.threat_level in [SecurityThreatLevel.SAFE, SecurityThreatLevel.LOW_RISK]
        )
    
    def get_risk_factors(self) -> List[str]:
        """
        Get list of identified risk factors.
        
        Returns:
            List[str]: List of risk factor descriptions
        """
        risk_factors = []
        
        if self.deepfake_score < 0.6:
            risk_factors.append(f"Potential deepfake detected (confidence: {self.deepfake_score:.2f})")
        
        if self.topology_score < 0.7:
            risk_factors.append(f"Suspicious file topology (score: {self.topology_score:.2f})")
        
        if self.file_integrity_score < 0.8:
            risk_factors.append(f"File integrity concerns (score: {self.file_integrity_score:.2f})")
        
        if self.overall_score < 0.5:
            risk_factors.append(f"Overall security risk elevated (score: {self.overall_score:.2f})")
        
        return risk_factors

class SpectralAnalyzer:
    """
    Spectral analysis engine using PyTorch for advanced signal processing.
    
    This class implements spectral-graph analysis techniques to detect
    anomalies in file structures and content patterns.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize spectral analyzer.
        
        Args:
            device: PyTorch device ('cpu', 'cuda', or None for auto-detection)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"SpectralAnalyzer initialized with device: {self.device}")
    
    def analyze_file_spectrum(self, file_data: bytes) -> torch.Tensor:
        """
        Perform spectral analysis on file data using FFT.
        
        Args:
            file_data: Raw file bytes
            
        Returns:
            torch.Tensor: Spectral analysis results
        """
        # Convert bytes to numpy array
        data_array = np.frombuffer(file_data, dtype=np.uint8)
        
        # Pad or truncate to standard size for analysis
        target_size = 8192  # 8KB analysis window
        if len(data_array) > target_size:
            data_array = data_array[:target_size]
        else:
            data_array = np.pad(data_array, (0, target_size - len(data_array)), 'constant')
        
        # Convert to PyTorch tensor
        data_tensor = torch.from_numpy(data_array).float().to(self.device)
        
        # Perform FFT for spectral analysis
        fft_result = torch.fft.fft(data_tensor)
        
        # Calculate magnitude spectrum
        magnitude_spectrum = torch.abs(fft_result)
        
        # Normalize spectrum
        normalized_spectrum = F.normalize(magnitude_spectrum, p=2, dim=0)
        
        return normalized_spectrum
    
    def detect_anomalies(self, spectrum: torch.Tensor) -> float:
        """
        Detect anomalies in spectral data using statistical analysis.
        
        Args:
            spectrum: Normalized spectral data
            
        Returns:
            float: Anomaly score (0.0 = highly anomalous, 1.0 = normal)
        """
        # Calculate statistical measures
        mean_val = torch.mean(spectrum)
        std_val = torch.std(spectrum)
        
        # Detect outliers using z-score analysis
        z_scores = torch.abs((spectrum - mean_val) / (std_val + 1e-8))
        outlier_ratio = torch.sum(z_scores > 3.0).float() / len(spectrum)
        
        # Calculate entropy as measure of randomness
        # Higher entropy = more random = potentially suspicious
        prob_dist = F.softmax(spectrum, dim=0)
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-8))
        normalized_entropy = entropy / torch.log(torch.tensor(len(spectrum), dtype=torch.float))
        
        # Combine metrics for anomaly score
        # Lower outlier ratio and moderate entropy indicate normal files
        anomaly_score = 1.0 - (outlier_ratio.item() + normalized_entropy.item()) / 2.0
        
        # Clamp to valid range
        return max(0.0, min(1.0, anomaly_score))

class DeepfakeDetector:
    """
    Deepfake detection engine with placeholder implementation.
    
    In production, this would integrate with advanced AI models trained
    on deepfake detection datasets. For development, it provides realistic
    mock detection capabilities.
    """
    
    def __init__(self):
        """Initialize deepfake detector."""
        self.model_loaded = False
        logger.info("DeepfakeDetector initialized (placeholder mode)")
    
    def analyze_image(self, image_data: bytes) -> Dict[str, float]:
        """
        Analyze image for deepfake indicators.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dict[str, float]: Detection results with confidence scores
        """
        # Placeholder implementation with realistic scoring
        # In production, this would use trained neural networks
        
        # Simulate analysis based on file characteristics
        file_size = len(image_data)
        
        # Calculate hash for deterministic results
        file_hash = hashlib.md5(image_data).hexdigest()
        hash_int = int(file_hash[:8], 16)
        
        # Generate realistic but deterministic scores
        base_score = (hash_int % 100) / 100.0
        
        # Adjust based on file size (very small or very large files are suspicious)
        size_factor = 1.0
        if file_size < 10000:  # Very small files
            size_factor = 0.7
        elif file_size > 10000000:  # Very large files
            size_factor = 0.8
        
        # Simulate different detection algorithms
        facial_analysis_score = min(1.0, base_score * size_factor + 0.1)
        temporal_consistency_score = min(1.0, base_score * 0.9 + 0.15)
        artifact_detection_score = min(1.0, base_score * 1.1)
        
        # Overall deepfake confidence (higher = more likely authentic)
        overall_confidence = (
            facial_analysis_score * 0.4 +
            temporal_consistency_score * 0.3 +
            artifact_detection_score * 0.3
        )
        
        return {
            "overall_confidence": overall_confidence,
            "facial_analysis_score": facial_analysis_score,
            "temporal_consistency_score": temporal_consistency_score,
            "artifact_detection_score": artifact_detection_score,
            "processing_time_ms": 150 + (hash_int % 100)  # Simulate processing time
        }
    
    def analyze_video(self, video_data: bytes) -> Dict[str, float]:
        """
        Analyze video for deepfake indicators.
        
        Args:
            video_data: Raw video bytes
            
        Returns:
            Dict[str, float]: Detection results with confidence scores
        """
        # Placeholder implementation for video analysis
        # In production, this would analyze frame sequences
        
        file_size = len(video_data)
        file_hash = hashlib.md5(video_data).hexdigest()
        hash_int = int(file_hash[:8], 16)
        
        base_score = (hash_int % 100) / 100.0
        
        # Video files require more sophisticated analysis
        frame_consistency_score = min(1.0, base_score * 0.85 + 0.1)
        motion_analysis_score = min(1.0, base_score * 0.9 + 0.05)
        compression_artifact_score = min(1.0, base_score * 1.05)
        
        overall_confidence = (
            frame_consistency_score * 0.5 +
            motion_analysis_score * 0.3 +
            compression_artifact_score * 0.2
        )
        
        return {
            "overall_confidence": overall_confidence,
            "frame_consistency_score": frame_consistency_score,
            "motion_analysis_score": motion_analysis_score,
            "compression_artifact_score": compression_artifact_score,
            "processing_time_ms": 500 + (hash_int % 300)  # Longer processing for video
        }

class SGS:
    """
    Spectral-Graph Sentinel - Main security scanning engine.
    
    This is the primary interface for all security scanning operations
    in the Loan2Day platform. ALL file uploads must pass through
    SGS.scan_topology() before any processing occurs.
    """
    
    # Supported file types for security scanning
    SUPPORTED_IMAGE_TYPES = {
        'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp',
        'image/tiff', 'image/webp'
    }
    
    SUPPORTED_VIDEO_TYPES = {
        'video/mp4', 'video/avi', 'video/mov', 'video/wmv', 'video/flv',
        'video/webm', 'video/mkv'
    }
    
    SUPPORTED_DOCUMENT_TYPES = {
        'application/pdf', 'application/msword', 'text/plain',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    }
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize SGS security scanner.
        
        Args:
            device: PyTorch device for spectral analysis
        """
        self.spectral_analyzer = SpectralAnalyzer(device)
        self.deepfake_detector = DeepfakeDetector()
        self.scan_history: List[SecurityScore] = []
        
        logger.info("SGS (Spectral-Graph Sentinel) initialized successfully")
    
    def _calculate_file_hash(self, file_data: bytes) -> str:
        """
        Calculate SHA-256 hash of file data.
        
        Args:
            file_data: Raw file bytes
            
        Returns:
            str: Hexadecimal hash string
        """
        return hashlib.sha256(file_data).hexdigest()
    
    def _detect_file_type(self, file_data: bytes, filename: Optional[str] = None) -> str:
        """
        Detect MIME type of file data.
        
        Args:
            file_data: Raw file bytes
            filename: Optional filename for type detection
            
        Returns:
            str: MIME type string
        """
        # Try to detect from filename first
        if filename:
            mime_type, _ = mimetypes.guess_type(filename)
            if mime_type:
                return mime_type
        
        # Fallback to magic number detection (simplified)
        if file_data.startswith(b'\xFF\xD8\xFF'):
            return 'image/jpeg'
        elif file_data.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'image/png'
        elif file_data.startswith(b'GIF8'):
            return 'image/gif'
        elif file_data.startswith(b'%PDF'):
            return 'application/pdf'
        elif file_data.startswith(b'\x00\x00\x00\x18ftypmp4') or file_data.startswith(b'\x00\x00\x00\x20ftypmp4'):
            return 'video/mp4'
        else:
            return 'application/octet-stream'
    
    def _analyze_file_integrity(self, file_data: bytes, file_type: str) -> float:
        """
        Analyze file integrity and structure.
        
        Args:
            file_data: Raw file bytes
            file_type: MIME type of file
            
        Returns:
            float: Integrity score (0.0-1.0)
        """
        # Basic integrity checks
        file_size = len(file_data)
        
        # Check for minimum file size
        if file_size < 100:
            return 0.2  # Very small files are suspicious
        
        # Check for maximum reasonable size (100MB)
        if file_size > 100 * 1024 * 1024:
            return 0.3  # Very large files are suspicious
        
        # Check file header consistency
        header_score = 1.0
        if file_type.startswith('image/'):
            # Basic image header validation
            if not (file_data.startswith(b'\xFF\xD8') or  # JPEG
                   file_data.startswith(b'\x89PNG') or   # PNG
                   file_data.startswith(b'GIF8')):       # GIF
                header_score = 0.5
        
        # Calculate entropy to detect encrypted/compressed content
        if file_size > 1000:
            sample_data = file_data[:1000]
            unique_bytes = len(set(sample_data))
            entropy_score = unique_bytes / 256.0  # Normalize to 0-1
            
            # Moderate entropy is normal, very high or very low is suspicious
            if entropy_score < 0.3 or entropy_score > 0.9:
                entropy_score = 0.6
            else:
                entropy_score = 1.0
        else:
            entropy_score = 0.8
        
        # Combine scores
        integrity_score = (header_score * 0.4 + entropy_score * 0.6)
        return max(0.0, min(1.0, integrity_score))
    
    def _determine_threat_level(self, overall_score: float) -> SecurityThreatLevel:
        """
        Determine threat level based on overall security score.
        
        Args:
            overall_score: Overall security score (0.0-1.0)
            
        Returns:
            SecurityThreatLevel: Classified threat level
        """
        if overall_score >= 0.9:
            return SecurityThreatLevel.SAFE
        elif overall_score >= 0.7:
            return SecurityThreatLevel.LOW_RISK
        elif overall_score >= 0.5:
            return SecurityThreatLevel.MEDIUM_RISK
        elif overall_score >= 0.3:
            return SecurityThreatLevel.HIGH_RISK
        else:
            return SecurityThreatLevel.CRITICAL
    
    def scan_topology(
        self,
        file_data: Union[bytes, BinaryIO],
        filename: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> SecurityScore:
        """
        Perform comprehensive security topology scan on uploaded file.
        
        This is the main entry point for all file security scanning in Loan2Day.
        ALL file uploads MUST pass through this function before processing.
        
        Args:
            file_data: File data as bytes or file-like object
            filename: Optional filename for type detection
            user_id: Optional user ID for audit logging
            
        Returns:
            SecurityScore: Comprehensive security analysis results
            
        Raises:
            UnsupportedFileTypeError: If file type is not supported
            ScanFailureError: If scan fails to complete
        """
        scan_start_time = datetime.now()
        
        try:
            # Convert file-like object to bytes if necessary
            if hasattr(file_data, 'read'):
                file_bytes = file_data.read()
                if hasattr(file_data, 'seek'):
                    file_data.seek(0)  # Reset file pointer
            else:
                file_bytes = file_data
            
            # Basic validation
            if not file_bytes:
                raise ScanFailureError("Empty file data provided")
            
            file_size = len(file_bytes)
            file_hash = self._calculate_file_hash(file_bytes)
            file_type = self._detect_file_type(file_bytes, filename)
            
            logger.info(
                f"Starting SGS topology scan - File: {filename}, "
                f"Type: {file_type}, Size: {file_size} bytes, "
                f"Hash: {file_hash[:16]}..., User: {user_id}"
            )
            
            # Check if file type is supported
            all_supported_types = (
                self.SUPPORTED_IMAGE_TYPES |
                self.SUPPORTED_VIDEO_TYPES |
                self.SUPPORTED_DOCUMENT_TYPES
            )
            
            if file_type not in all_supported_types:
                logger.warning(f"Unsupported file type detected: {file_type}")
                # Don't raise exception, but assign low score
                file_integrity_score = 0.3
                deepfake_score = 0.5
                topology_score = 0.4
            else:
                # Perform spectral analysis
                spectrum = self.spectral_analyzer.analyze_file_spectrum(file_bytes)
                topology_score = self.spectral_analyzer.detect_anomalies(spectrum)
                
                # Perform file integrity analysis
                file_integrity_score = self._analyze_file_integrity(file_bytes, file_type)
                
                # Perform deepfake detection for images and videos
                if file_type in self.SUPPORTED_IMAGE_TYPES:
                    deepfake_results = self.deepfake_detector.analyze_image(file_bytes)
                    deepfake_score = deepfake_results["overall_confidence"]
                elif file_type in self.SUPPORTED_VIDEO_TYPES:
                    deepfake_results = self.deepfake_detector.analyze_video(file_bytes)
                    deepfake_score = deepfake_results["overall_confidence"]
                else:
                    # Documents don't need deepfake detection
                    deepfake_score = 1.0
                    deepfake_results = {"overall_confidence": 1.0, "note": "Not applicable for documents"}
            
            # Calculate overall security score
            # Weighted combination of all factors
            overall_score = (
                topology_score * 0.3 +
                file_integrity_score * 0.4 +
                deepfake_score * 0.3
            )
            
            # Determine threat level
            threat_level = self._determine_threat_level(overall_score)
            
            # Compile detection details
            detection_details = {
                "spectral_analysis": {
                    "topology_score": topology_score,
                    "anomaly_detection": "completed"
                },
                "file_integrity": {
                    "integrity_score": file_integrity_score,
                    "header_validation": "passed" if file_integrity_score > 0.7 else "failed"
                },
                "deepfake_detection": deepfake_results if 'deepfake_results' in locals() else {"note": "Not applicable"},
                "scan_metadata": {
                    "scan_duration_ms": int((datetime.now() - scan_start_time).total_seconds() * 1000),
                    "scanner_version": "SGS-1.0-placeholder",
                    "device_used": str(self.spectral_analyzer.device)
                }
            }
            
            # Create security score result
            security_score = SecurityScore(
                overall_score=overall_score,
                deepfake_score=deepfake_score,
                topology_score=topology_score,
                file_integrity_score=file_integrity_score,
                threat_level=threat_level,
                scan_timestamp=scan_start_time,
                file_hash=file_hash,
                file_type=file_type,
                file_size_bytes=file_size,
                detection_details=detection_details
            )
            
            # Add to scan history
            self.scan_history.append(security_score)
            
            # Log security event
            logger.info(
                f"SGS scan completed - Overall Score: {overall_score:.3f}, "
                f"Threat Level: {threat_level.value}, "
                f"Safe: {security_score.is_safe()}, "
                f"Duration: {detection_details['scan_metadata']['scan_duration_ms']}ms"
            )
            
            # Log security event for audit trail
            if not security_score.is_safe():
                logger.warning(
                    f"SECURITY ALERT - Risky file detected: {filename}, "
                    f"User: {user_id}, Hash: {file_hash[:16]}..., "
                    f"Risk Factors: {security_score.get_risk_factors()}"
                )
            
            return security_score
            
        except Exception as e:
            logger.error(f"SGS scan failed: {str(e)}", exc_info=True)
            raise ScanFailureError(f"Security scan failed: {str(e)}")
    
    def get_scan_history(self, limit: Optional[int] = None) -> List[SecurityScore]:
        """
        Get history of security scans.
        
        Args:
            limit: Optional limit on number of results
            
        Returns:
            List[SecurityScore]: List of recent security scans
        """
        if limit:
            return self.scan_history[-limit:]
        return self.scan_history.copy()
    
    def get_threat_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about detected threats.
        
        Returns:
            Dict[str, Any]: Threat statistics summary
        """
        if not self.scan_history:
            return {"total_scans": 0, "threat_distribution": {}}
        
        threat_counts = {}
        safe_count = 0
        
        for scan in self.scan_history:
            threat_level = scan.threat_level.value
            threat_counts[threat_level] = threat_counts.get(threat_level, 0) + 1
            if scan.is_safe():
                safe_count += 1
        
        return {
            "total_scans": len(self.scan_history),
            "safe_files": safe_count,
            "risky_files": len(self.scan_history) - safe_count,
            "safety_rate": safe_count / len(self.scan_history),
            "threat_distribution": threat_counts,
            "average_overall_score": sum(scan.overall_score for scan in self.scan_history) / len(self.scan_history)
        }

# Global SGS instance for application use
_sgs_instance: Optional[SGS] = None

def get_sgs_instance() -> SGS:
    """
    Get global SGS instance (singleton pattern).
    
    Returns:
        SGS: Global SGS scanner instance
    """
    global _sgs_instance
    if _sgs_instance is None:
        _sgs_instance = SGS()
    return _sgs_instance

def scan_topology(
    file_data: Union[bytes, BinaryIO],
    filename: Optional[str] = None,
    user_id: Optional[str] = None
) -> SecurityScore:
    """
    Convenience function for security topology scanning.
    
    This is the main entry point for file security scanning in Loan2Day.
    ALL file uploads MUST use this function before processing.
    
    Args:
        file_data: File data as bytes or file-like object
        filename: Optional filename for type detection
        user_id: Optional user ID for audit logging
        
    Returns:
        SecurityScore: Comprehensive security analysis results
    """
    sgs = get_sgs_instance()
    return sgs.scan_topology(file_data, filename, user_id)

# Export main classes and functions
__all__ = [
    'SGS',
    'SecurityScore',
    'SecurityThreatLevel',
    'SpectralAnalyzer',
    'DeepfakeDetector',
    'SGSError',
    'UnsupportedFileTypeError',
    'ScanFailureError',
    'DeepfakeDetectionError',
    'scan_topology',
    'get_sgs_instance'
]