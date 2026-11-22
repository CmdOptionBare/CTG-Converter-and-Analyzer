"""Lightweight CPU/ONNX feature descriptors and matchers for overlap verification."""

from .onnx_matcher import OnnxPatchMatcher, ensure_tiny_onnx_descriptor_model

__all__ = ["OnnxPatchMatcher", "ensure_tiny_onnx_descriptor_model"]
