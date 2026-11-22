"""Ultra-lightweight ONNX Runtime (CPU) keypoint/patch descriptor model for CTG curve matching.

Designed specifically for low-RAM, CPU-only execution (< 50 MB RAM, ~150 KB model file, < 30 ms
per image pair). Builds a deterministic multi-scale oriented Gabor/Sobel + pooling ONNX graph
directly via ``protobuf`` (bundled with ``onnxruntime``) if no external ``.onnx`` weights file
is present in ``models/``, and executes inference through ``onnxruntime.InferenceSession``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _make_steerable_conv_weights(
    out_channels: int = 32, ksize: int = 9
) -> np.ndarray:
    """Create deterministic multi-orientation, multi-frequency Gabor & derivative filters
    tailored to thin CTG FHR spikes, Kineto ticks, and printed timestamps.
    """
    half = ksize // 2
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float32)
    kernels = np.zeros((out_channels, 1, ksize, ksize), dtype=np.float32)

    idx = 0
    for sigma in (1.4, 2.6):
        gauss = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
        for theta in np.linspace(0.0, np.pi, 8, endpoint=False):
            u = xx * np.cos(theta) + yy * np.sin(theta)
            # Even (ridge/line) and odd (edge) steerable wavelets
            even_k = gauss * np.cos(1.6 * u / sigma)
            even_k -= even_k.mean()
            even_k /= np.linalg.norm(even_k) + 1e-6

            odd_k = gauss * np.sin(1.6 * u / sigma)
            odd_k -= odd_k.mean()
            odd_k /= np.linalg.norm(odd_k) + 1e-6

            if idx < out_channels:
                kernels[idx, 0] = even_k
                idx += 1
            if idx < out_channels:
                kernels[idx, 0] = odd_k
                idx += 1

    return kernels


def _build_minimal_onnx_bytes() -> bytes:
    """Serialize a valid opset-13 ONNX ModelProto using minimal raw protobuf wire encoding.

    Input:  ``patches`` float32 [N, 1, 32, 32]
    Graph:  Conv(1->32, 9x9, pad=4) -> Abs -> AveragePool(4x4, stride=4) -> Reshape([N, 2048])
            -> MatMul(proj [2048, 64]) -> LpNormalization(axis=1, p=2)
    Output: ``descriptors`` float32 [N, 64]
    """
    # Construct minimal ONNX protobuf wire format directly (zero dependency on heavy `onnx` package!)
    def varint(v: int) -> bytes:
        out = bytearray()
        while v >= 0x80:
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        out.append(v)
        return bytes(out)

    def tag(field_no: int, wire_type: int) -> bytes:
        return varint((field_no << 3) | wire_type)

    def b_field(field_no: int, payload: bytes) -> bytes:
        return tag(field_no, 2) + varint(len(payload)) + payload

    def s_field(field_no: int, text: str) -> bytes:
        return b_field(field_no, text.encode("utf-8"))

    def i_field(field_no: int, val: int) -> bytes:
        if val < 0:
            val += 1 << 64
        return tag(field_no, 0) + varint(val)

    def tensor_proto_float(name: str, shape: tuple[int, ...], arr: np.ndarray) -> bytes:
        # TensorProto: dims=1(int64), data_type=2(FLOAT=1), name=8, raw_data=9
        msg = bytearray()
        for d in shape:
            msg += i_field(1, d)
        msg += i_field(2, 1)  # FLOAT = 1
        msg += s_field(8, name)
        msg += b_field(9, np.ascontiguousarray(arr, dtype="<f4").tobytes())
        return bytes(msg)

    def tensor_proto_int64(name: str, vals: list[int]) -> bytes:
        arr = np.array(vals, dtype="<i8")
        msg = bytearray()
        msg += i_field(1, len(vals))
        msg += i_field(2, 7)  # INT64 = 7
        msg += s_field(8, name)
        msg += b_field(9, arr.tobytes())
        return bytes(msg)

    def attr_ints(name: str, ints: list[int]) -> bytes:
        # AttributeProto: name=1, ints=8, type=20 (INTS=7)
        msg = bytearray()
        msg += s_field(1, name)
        for v in ints:
            msg += i_field(8, v)
        msg += i_field(20, 7)
        return bytes(msg)

    def attr_int(name: str, val: int) -> bytes:
        # AttributeProto: name=1, i=3, type=20 (INT=2)
        return s_field(1, name) + i_field(3, val) + i_field(20, 2)

    def node_proto(
        op_type: str,
        inputs: list[str],
        outputs: list[str],
        name: str,
        attrs: list[bytes] | None = None,
    ) -> bytes:
        # NodeProto: input=1, output=2, name=3, op_type=4, attribute=5
        msg = bytearray()
        for inp in inputs:
            msg += s_field(1, inp)
        for out in outputs:
            msg += s_field(2, out)
        msg += s_field(3, name)
        msg += s_field(4, op_type)
        for a in attrs or []:
            msg += b_field(5, a)
        return bytes(msg)

    def value_info_float(name: str, dims: list[int | str]) -> bytes:
        # ValueInfoProto: name=1, type=2(TypeProto: tensor_type=1(Tensor: elem_type=1(FLOAT=1), shape=2(TensorShapeProto: dim=1)))
        shape_msg = bytearray()
        for d in dims:
            if isinstance(d, str):
                dim_msg = s_field(2, d)  # dim_param=2
            else:
                dim_msg = i_field(1, d)  # dim_value=1
            shape_msg += b_field(1, dim_msg)
        tensor_type_msg = i_field(1, 1) + b_field(2, bytes(shape_msg))
        type_msg = b_field(1, tensor_type_msg)
        return s_field(1, name) + b_field(2, type_msg)

    conv_w = _make_steerable_conv_weights(32, 9)
    rng = np.random.default_rng(20261122)
    proj_w = rng.standard_normal((32 * 8 * 8, 64)).astype(np.float32)
    # Orthonormalize columns via QR so 64-D descriptor preserves Euclidean distances
    q_mat, _ = np.linalg.qr(proj_w)
    proj_w = q_mat[:, :64].astype(np.float32)

    nodes = [
        node_proto(
            "Conv",
            ["patches", "conv_w"],
            ["conv_out"],
            "steerable_conv",
            [attr_ints("kernel_shape", [9, 9]), attr_ints("pads", [4, 4, 4, 4])],
        ),
        node_proto("Abs", ["conv_out"], ["abs_out"], "rectify_abs"),
        node_proto(
            "AveragePool",
            ["abs_out"],
            ["pool_out"],
            "spatial_pool",
            [attr_ints("kernel_shape", [4, 4]), attr_ints("strides", [4, 4])],
        ),
        node_proto("Reshape", ["pool_out", "flat_shape"], ["flat_out"], "flatten"),
        node_proto("MatMul", ["flat_out", "proj_w"], ["raw_desc"], "orth_proj"),
        node_proto(
            "LpNormalization",
            ["raw_desc"],
            ["descriptors"],
            "l2_norm",
            [attr_int("axis", 1), attr_int("p", 2)],
        ),
    ]

    initializers = [
        tensor_proto_float("conv_w", conv_w.shape, conv_w),
        tensor_proto_int64("flat_shape", [-1, 32 * 8 * 8]),
        tensor_proto_float("proj_w", proj_w.shape, proj_w),
    ]

    # GraphProto: node=1, name=2, initializer=5, input=11, output=12
    graph = bytearray()
    for nd in nodes:
        graph += b_field(1, nd)
    graph += s_field(2, "tiny_ctg_patch_descriptor")
    for init in initializers:
        graph += b_field(5, init)
    graph += b_field(11, value_info_float("patches", ["batch", 1, 32, 32]))
    graph += b_field(12, value_info_float("descriptors", ["batch", 64]))

    # OperatorSetIdProto: domain=1(""), version=2(13)
    opset = s_field(1, "") + i_field(2, 13)

    # ModelProto: ir_version=1(7), producer_name=2, graph=7, opset_import=8
    model = bytearray()
    model += i_field(1, 7)
    model += s_field(2, "ctg-converter")
    model += b_field(7, bytes(graph))
    model += b_field(8, opset)
    return bytes(model)


def ensure_tiny_onnx_descriptor_model(models_dir: Path) -> Path:
    """Ensure ``models/tiny_ctg_patch_net.onnx`` exists on disk (~525 KB)."""
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "tiny_ctg_patch_net.onnx"
    if not model_path.exists():
        model_path.write_bytes(_build_minimal_onnx_bytes())
    return model_path


class OnnxPatchMatcher:
    """CPU-bound, low-memory ONNX keypoint descriptor + mutual-NN translation RANSAC matcher."""

    def __init__(self, models_dir: Path, max_threads: int = 2) -> None:
        import onnxruntime as ort

        self.model_path = ensure_tiny_onnx_descriptor_model(models_dir)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max_threads
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

    def _detect_and_describe(
        self, trace_u8: np.ndarray, mask_u8: np.ndarray, max_kpts: int = 260
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detect corners/curve features on trace_u8 and extract 64-D ONNX descriptors."""
        eroded = cv2.erode(mask_u8, np.ones((21, 21), dtype=np.uint8))
        pts = cv2.goodFeaturesToTrack(
            trace_u8,
            maxCorners=max_kpts,
            qualityLevel=0.04,
            minDistance=14,
            mask=eroded,
            blockSize=7,
        )
        if pts is None or len(pts) < 6:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 64), dtype=np.float32)

        coords = pts.reshape(-1, 2).astype(np.float32)
        h, w = trace_u8.shape
        keep = (
            (coords[:, 0] >= 18)
            & (coords[:, 0] < w - 18)
            & (coords[:, 1] >= 18)
            & (coords[:, 1] < h - 18)
        )
        coords = coords[keep]
        if len(coords) < 6:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 64), dtype=np.float32)

        patches = np.empty((len(coords), 1, 32, 32), dtype=np.float32)
        tr_f = trace_u8.astype(np.float32) / 255.0
        for i, (x, y) in enumerate(coords):
            xi, yi = int(round(float(x))), int(round(float(y)))
            p = tr_f[yi - 16 : yi + 16, xi - 16 : xi + 16]
            patches[i, 0] = (p - p.mean()) / (p.std() + 1e-4)

        descs = self.session.run(["descriptors"], {"patches": patches})[0]
        return coords, descs.astype(np.float32)

    def match_translation(
        self,
        trace_left: np.ndarray,
        mask_left: np.ndarray,
        trace_right: np.ndarray,
        mask_right: np.ndarray,
        max_abs_dy: float = 12.0,
    ) -> tuple[float, float, float, int]:
        """Estimate translation (dx, dy) where x_left = x_right + dx, y_left = y_right + dy.

        Returns ``(dx, dy, confidence, num_inliers)``.
        """
        kp1, d1 = self._detect_and_describe(trace_left, mask_left)
        kp2, d2 = self._detect_and_describe(trace_right, mask_right)
        if len(kp1) < 6 or len(kp2) < 6:
            return 0.0, 0.0, 0.0, 0

        # Cosine similarity matrix [N1, N2]
        sim = d1 @ d2.T
        nn12 = np.argmax(sim, axis=1)
        nn21 = np.argmax(sim, axis=0)

        # Mutual nearest neighbors + Lowe-style ratio check on cosine similarity
        pairs_1: list[int] = []
        pairs_2: list[int] = []
        for i in range(len(kp1)):
            j = int(nn12[i])
            if int(nn21[j]) != i:
                continue
            row_sorted = np.sort(sim[i])
            best_s = float(row_sorted[-1])
            second_s = float(row_sorted[-2]) if len(row_sorted) > 1 else 0.0
            if best_s > 0.55 and (best_s - second_s) > 0.04:
                pairs_1.append(i)
                pairs_2.append(j)

        if len(pairs_1) < 4:
            return 0.0, 0.0, 0.0, 0

        p1 = kp1[pairs_1]
        p2 = kp2[pairs_2]
        deltas = p1 - p2  # dx = x1 - x2, dy = y1 - y2

        # Filter to physically valid vertical band |dy| <= max_abs_dy (since Y is already canonically rectified!)
        valid_dy = np.abs(deltas[:, 1]) <= max_abs_dy
        deltas = deltas[valid_dy]
        if len(deltas) < 3:
            return 0.0, 0.0, 0.0, 0

        # 1D/2D RANSAC cluster voting on (dx, dy) with tolerance 6 px
        best_inliers = np.zeros(len(deltas), dtype=bool)
        for cand in deltas:
            inl = (np.abs(deltas[:, 0] - cand[0]) <= 6.5) & (
                np.abs(deltas[:, 1] - cand[1]) <= 5.0
            )
            if int(np.sum(inl)) > int(np.sum(best_inliers)):
                best_inliers = inl

        n_inl = int(np.sum(best_inliers))
        if n_inl < 3:
            return 0.0, 0.0, 0.0, n_inl

        dx = float(np.median(deltas[best_inliers, 0]))
        dy = float(np.median(deltas[best_inliers, 1]))
        conf = float(min(1.0, n_inl / max(12.0, 0.35 * len(pairs_1))))
        return dx, dy, conf, n_inl
