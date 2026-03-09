#!/usr/bin/env python3
"""
Linear stream via V4L2 directly — bypasses Argus/EGL entirely.
Works with IMX477 (RPi HQ cam) on Jetson Nano.

Run:
    python3 linear_stream_v4l2.py
    python3 linear_stream_v4l2.py --device /dev/video1
    python3 linear_stream_v4l2.py --width 1920 --height 1080
"""

import cv2
import numpy as np
import ctypes
import fcntl
import mmap
import struct
import argparse

# ── V4L2 constants ───────────────────────────────────────────
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP             = 1
V4L2_PIX_FMT_SRGGB10P       = 0x70524752  # 'pRGG' — packed RG10
VIDIOC_QUERYCAP   = 0x80685600
VIDIOC_S_FMT      = 0xC0D05605
VIDIOC_REQBUFS    = 0xC0145608
VIDIOC_QUERYBUF   = 0xC0445609
VIDIOC_QBUF       = 0xC044560F
VIDIOC_DQBUF      = 0xC0445611
VIDIOC_STREAMON   = 0x40045612
VIDIOC_STREAMOFF  = 0x40045613


def unpack_rg10(raw: bytes, width: int, height: int) -> np.ndarray:
    """
    Unpack 10-bit packed Bayer (RG10 / SRGGB10P) to uint16.
    Every 5 bytes encodes 4 pixels: [p0_hi, p1_hi, p2_hi, p3_hi, lsbs]
    """
    data = np.frombuffer(raw, dtype=np.uint8)
    # Trim to exact size (stride may add padding)
    stride = (width * 10 + 7) // 8   # bytes per row (packed)
    rows = []
    for r in range(height):
        row = data[r * stride : r * stride + stride]
        # Each group of 5 bytes = 4 pixels
        groups = row[: (len(row) // 5) * 5].reshape(-1, 5)
        px = np.zeros((groups.shape[0], 4), dtype=np.uint16)
        px[:, 0] = (groups[:, 0].astype(np.uint16) << 2) | ((groups[:, 4]     ) & 0x03)
        px[:, 1] = (groups[:, 1].astype(np.uint16) << 2) | ((groups[:, 4] >> 2) & 0x03)
        px[:, 2] = (groups[:, 2].astype(np.uint16) << 2) | ((groups[:, 4] >> 4) & 0x03)
        px[:, 3] = (groups[:, 3].astype(np.uint16) << 2) | ((groups[:, 4] >> 6) & 0x03)
        rows.append(px.reshape(-1))
    bayer = np.array(rows, dtype=np.uint16)[:, :width]
    return bayer


def bayer_to_bgr(bayer16: np.ndarray) -> np.ndarray:
    """
    Debayer a linear uint16 RGGB Bayer image → linear uint8 BGR.
    No gamma applied — output is linear light.
    """
    # Normalize 10-bit → 8-bit linearly (no gamma)
    bayer8 = (bayer16 >> 2).astype(np.uint8)
    # OpenCV debayer — RGGB pattern
    bgr = cv2.cvtColor(bayer8, cv2.COLOR_BayerRG2BGR)
    return bgr


class V4L2Camera:
    def __init__(self, device, width, height):
        self.fd = open(device, "rb+", buffering=0)
        self.width  = width
        self.height = height
        self.buffers = []
        self._set_format()
        self._request_buffers()
        self._map_buffers()
        self._queue_all()

    def _ioctl(self, req, buf):
        return fcntl.ioctl(self.fd, req, buf)

    def _set_format(self):
        # struct v4l2_format (simplified for Video Capture)
        # type(4) + pix: width(4) height(4) pixelformat(4) field(4)
        #                bytesperline(4) sizeimage(4) colorspace(4) ...
        fmt = struct.pack("III IIIIII 200x",
            V4L2_BUF_TYPE_VIDEO_CAPTURE,
            0, 0,  # padding
            self.width,
            self.height,
            V4L2_PIX_FMT_SRGGB10P,
            0, 0, 0, 0
        )
        try:
            self._ioctl(VIDIOC_S_FMT, bytearray(fmt))
            print(f"Format set: {self.width}x{self.height} RG10")
        except Exception as e:
            print(f"[WARN] S_FMT ioctl failed ({e}), continuing anyway")

    def _request_buffers(self, count=4):
        # struct v4l2_requestbuffers: count(4) type(4) memory(4) reserved(8)
        req = struct.pack("III8x", count,
                          V4L2_BUF_TYPE_VIDEO_CAPTURE,
                          V4L2_MEMORY_MMAP)
        result = bytearray(req)
        self._ioctl(VIDIOC_REQBUFS, result)
        n = struct.unpack("I", result[:4])[0]
        print(f"Allocated {n} buffers")

    def _map_buffers(self, count=4):
        for i in range(count):
            # struct v4l2_buffer (simplified): type(4) memory(4) index(4) ...
            buf = bytearray(struct.pack(
                "IIIIIIIII4xQIi4x",
                V4L2_BUF_TYPE_VIDEO_CAPTURE,  # type
                V4L2_MEMORY_MMAP,             # memory
                i,                            # index
                0, 0, 0, 0, 0, 0, 0, 0, 0    # rest zeroed
            ))
            try:
                self._ioctl(VIDIOC_QUERYBUF, buf)
                length = struct.unpack_from("I", buf, 32)[0]
                offset = struct.unpack_from("I", buf, 40)[0]
                mm = mmap.mmap(self.fd.fileno(), length,
                               mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
                               offset=offset)
                self.buffers.append((mm, length))
            except Exception as e:
                print(f"[WARN] Buffer {i} map failed: {e}")

    def _queue_all(self):
        for i in range(len(self.buffers)):
            self._queue(i)

    def _queue(self, index):
        buf = bytearray(struct.pack(
            "IIIIIIIII4xQIi4x",
            V4L2_BUF_TYPE_VIDEO_CAPTURE,
            V4L2_MEMORY_MMAP,
            index,
            0, 0, 0, 0, 0, 0, 0, 0, 0
        ))
        self._ioctl(VIDIOC_QBUF, buf)

    def start(self):
        buf_type = struct.pack("I", V4L2_BUF_TYPE_VIDEO_CAPTURE)
        self._ioctl(VIDIOC_STREAMON, bytearray(buf_type))
        print("Stream started")

    def read(self):
        # Dequeue
        buf = bytearray(struct.pack(
            "IIIIIIIII4xQIi4x",
            V4L2_BUF_TYPE_VIDEO_CAPTURE,
            V4L2_MEMORY_MMAP,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ))
        self._ioctl(VIDIOC_DQBUF, buf)
        index    = struct.unpack_from("I", buf, 8)[0]
        bytesused = struct.unpack_from("I", buf, 20)[0]
        mm, _ = self.buffers[index]
        mm.seek(0)
        data = mm.read(bytesused)
        # Re-queue
        self._queue(index)
        return data

    def stop(self):
        buf_type = struct.pack("I", V4L2_BUF_TYPE_VIDEO_CAPTURE)
        self._ioctl(VIDIOC_STREAMOFF, bytearray(buf_type))

    def close(self):
        self.stop()
        for mm, _ in self.buffers:
            mm.close()
        self.fd.close()


# ── Main ─────────────────────────────────────────────────────
def main(args):
    print(f"Opening {args.device} at {args.width}x{args.height}")
    cam = V4L2Camera(args.device, args.width, args.height)
    cam.start()

    print("Streaming — press Q to quit, S to save frame")

    while True:
        raw = cam.read()
        bayer = unpack_rg10(raw, args.width, args.height)
        bgr   = bayer_to_bgr(bayer)   # linear, no gamma

        # Show a scaled-down preview
        preview = cv2.resize(bgr, (960, 540))
        cv2.imshow("Linear Stream (V4L2 direct)", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            cv2.imwrite("linear_frame.png", bgr)
            print("Saved linear_frame.png")

    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width",  type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    main(args)