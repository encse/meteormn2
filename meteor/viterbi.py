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

    BLOCK_BITS = 1024
    BLOCK_SOFT = BLOCK_BITS * 2

    def __init__(self):

        gr.basic_block.__init__(
            self,
            name="viterbi",
            in_sig=[np.complex64],
            out_sig=[np.uint8, np.float32],
        )

        polys = [109,79]

        self.iq_interleaved = np.zeros(self.BLOCK_SOFT, dtype=np.float32)

        self.dec = fec.cc_decoder.make(
            self.BLOCK_BITS,7,2,polys,0,-1,fec.CC_STREAMING,False
        )

        self.enc = fec.cc_encoder.make(
            self.BLOCK_BITS,7,2,polys,0,fec.CC_STREAMING,False
        )
        self.history_overlap = int(self.dec.get_history())

        if self.history_overlap < 0 or self.history_overlap % 2 != 0:
            raise RuntimeError("Decoder history is invalid")

        self.history_iq = self.history_overlap // 2

        # buffers

        self.prev_iq = np.zeros(self.history_iq, dtype=np.complex64)
        self.full_iq = np.zeros(self.history_iq + self.BLOCK_BITS, dtype=np.complex64)

        self.soft_float = np.zeros(self.history_overlap + self.BLOCK_SOFT, dtype=np.float32)
        self.soft_u8 = np.zeros(self.history_overlap + self.BLOCK_SOFT, dtype=np.uint8)

        self.decoded_a = np.zeros(self.history_iq + self.BLOCK_BITS, dtype=np.uint8)
        self.decoded_b = np.zeros(self.history_iq + self.BLOCK_BITS, dtype=np.uint8)

        self.reencoded = np.zeros(self.history_overlap + self.BLOCK_SOFT, dtype=np.uint8)

       
        # capsules
        self.soft_caps = ndarray_to_capsule(self.soft_u8)
        self.decoded_a_caps = ndarray_to_capsule(self.decoded_a)
        self.decoded_b_caps = ndarray_to_capsule(self.decoded_b)
        self.renc_caps = ndarray_to_capsule(self.reencoded)

    def build_soft_input(self, iq_block, multiply_by_j):
        if self.history_iq > 0:
            self.full_iq[:self.history_iq] = self.prev_iq

        self.full_iq[self.history_iq:] = iq_block

        if multiply_by_j:
            view = self.full_iq * np.complex64(1j)
        else:
            view = self.full_iq

        self.soft_float[0::2] = view.real
        self.soft_float[1::2] = view.imag

    def float_to_soft(self):

        scaled = np.rint(self.soft_float * 127.0 + 128.0)
        scaled = np.clip(scaled, 0, 255)
        self.soft_u8[:] = scaled.astype(np.uint8)

    def decode_and_measure(self, decoded_caps):
        self.float_to_soft()
        self.dec.generic_work(self.soft_caps, decoded_caps)
        self.enc.generic_work(decoded_caps, self.renc_caps)
        return self.compute_ber()
    
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
        iq_in = input_items[0]
        bits_out = output_items[0]
        ber_out = output_items[1]

        if len(iq_in) < self.BLOCK_BITS:
            return 0

        if len(bits_out) < self.BLOCK_BITS:
            return 0

        if len(ber_out) < self.BLOCK_BITS:
            return 0

        iq_block = iq_in[:self.BLOCK_BITS]

        self.build_soft_input(iq_block, multiply_by_j=False)
        ber_a = self.decode_and_measure(self.decoded_a_caps)

        self.build_soft_input(iq_block, multiply_by_j=True)
        ber_b = self.decode_and_measure(self.decoded_b_caps)

        if ber_b < ber_a:
            bits_out[:self.BLOCK_BITS] = self.decoded_b[-self.BLOCK_BITS:]
            ber_out[:self.BLOCK_BITS] = ber_b
        else:
            bits_out[:self.BLOCK_BITS] = self.decoded_a[-self.BLOCK_BITS:]
            ber_out[:self.BLOCK_BITS] = ber_a

        if self.history_iq > 0:
            self.prev_iq[:] = iq_block[-self.history_iq:]

        self.consume(0, self.BLOCK_BITS)
        return self.BLOCK_BITS