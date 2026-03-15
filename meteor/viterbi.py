import ctypes
import numpy as np
from gnuradio import gr, fec


def ndarray_to_capsule(arr):
    """
    Wrap a contiguous numpy array data pointer into a PyCapsule.
    The numpy array must stay alive while the capsule is used.
    """
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError("Array must be C-contiguous")

    ptr = ctypes.c_void_p(arr.ctypes.data)
    pycapsule_new = ctypes.pythonapi.PyCapsule_New
    pycapsule_new.restype = ctypes.py_object
    pycapsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return pycapsule_new(ptr, None, None)

class Viterbi(gr.basic_block):

    BLOCK_SOFT = 2048
    BLOCK_BITS = BLOCK_SOFT // 2

    def __init__(self):

        gr.basic_block.__init__(
            self,
            name="viterbi",
            in_sig=[np.float32],
            out_sig=[np.uint8, np.float32],
        )

        self.set_output_multiple(self.BLOCK_BITS)

        polys = [109,79]

        self.dec = fec.cc_decoder.make(
            self.BLOCK_BITS,7,2,polys,0,-1,fec.CC_STREAMING,False
        )

        self.enc = fec.cc_encoder.make(
            self.BLOCK_BITS,7,2,polys,0,fec.CC_STREAMING,False
        )
        self.history_overlap = int(self.dec.get_history())

        self.history_overlap = 0

        if self.history_overlap < 0 or self.history_overlap % 2 != 0:
            raise RuntimeError("Decoder history is invalid")

        # buffers

        self.soft_float = np.zeros(self.history_overlap + self.BLOCK_SOFT, dtype=np.float32)
        self.soft_u8 = np.zeros(self.history_overlap + self.BLOCK_SOFT, dtype=np.uint8)

        self.decoded = np.zeros(self.history_overlap // 2 + self.BLOCK_BITS, dtype=np.uint8)
        self.reencoded = np.zeros(self.history_overlap + self.BLOCK_SOFT, dtype=np.uint8)

        # capsules

        self.soft_caps = ndarray_to_capsule(self.soft_u8)
        self.dec_caps = ndarray_to_capsule(self.decoded)
        self.renc_caps = ndarray_to_capsule(self.reencoded)

        self.prev_soft_float = np.zeros(self.history_overlap, dtype=np.float32)

    def float_to_soft(self):

        scaled = np.rint(self.soft_float * 127.0 + 128.0)
        scaled = np.clip(scaled, 0, 255)
        self.soft_u8[:] = scaled.astype(np.uint8)

    def compute_ber(self):

        raw = self.soft_u8

        mask = raw != 128
        total = int(mask.sum())

        if total == 0:
            return 10.0

        hard = (raw > 127).astype(np.uint8)

        errors = int((hard[mask] != self.reencoded[mask]).sum())

        return float(errors) / float(total) * 2.5

    def general_work(self, input_items, output_items):

        soft_in = input_items[0]
        bits_out = output_items[0]
        ber_out = output_items[1]

        if len(soft_in) < self.BLOCK_SOFT:
            return 0

        if len(bits_out) < self.BLOCK_BITS:
            return 0

        if len(ber_out) < self.BLOCK_BITS:
            return 0

        # Build one decoder work block:
        # [previous history overlap | new input samples]
        self.soft_float[:self.history_overlap] = self.prev_soft_float
        self.soft_float[self.history_overlap:] = soft_in[:self.BLOCK_SOFT]

        # Convert signed float soft samples to unsigned uchar soft samples
        self.float_to_soft()

        n = self.dec.generic_work(
            ndarray_to_capsule(self.soft_u8),
            ndarray_to_capsule(self.decoded)
        )

        # Re-encode the decoded bits back to coded form
        self.enc.generic_work(
            ndarray_to_capsule(self.decoded),
            ndarray_to_capsule(self.reencoded)
        )

        # Compute BER on the same work block
        ber = self.compute_ber()

        # Publish outputs
        bits_out[:self.BLOCK_BITS] = self.decoded[-self.BLOCK_BITS:]
        ber_out[:self.BLOCK_BITS] = ber

        # Save the tail of the current block for the next call
        if self.history_overlap > 0:
            self.prev_soft_float[:] = self.soft_float[-self.history_overlap:]

        self.consume(0, self.BLOCK_SOFT)

        return self.BLOCK_BITS