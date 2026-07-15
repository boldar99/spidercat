import math
import numpy as np
from collections import deque

from tqdm import tqdm


class LutDecoder:
    def __init__(self, H: np.ndarray, max_decodable_weight=None, verbose=False):
        self.H = H
        self.verbose = verbose
        self.m, self.n = self.H.shape
        self.max_weight = max_decodable_weight
        if self.max_weight is None:
            self.max_weight = self.n  # Unbounded max weight

        # Use dictionary if lut_size is too large to fit safely in memory
        self.use_dict = self.m > 25
        dtype = object if self.use_dict else np.int64
        self.powers_of_2 = (1 << np.arange(self.m, dtype=dtype)[::-1])
        self.packed_len = math.ceil(self.n / 8)

        if verbose:
            print(f"LUT Using {'dict' if self.use_dict else 'np.ndarray'} for 2^{self.m} = {1 << self.m} total syndromes")

        
        if self.use_dict:
            self.lut_size = float('inf')  # Unbounded for dicts
            self.dict_lut = {}
            self.can_correct_set = set()
        else:
            self.lut_size = 1 << self.m
            self.lut = np.zeros((self.lut_size, self.packed_len), dtype=np.uint8)
            self.can_correct = np.zeros(math.ceil(self.lut_size / 8), dtype=np.uint8)
            
        self._build_table()

    def _syndrome_int(self, e):
        s = (e @ self.H.T) % 2
        return s @ self.powers_of_2

    def _build_table(self):
        e_zero = np.zeros(self.n, dtype=np.bool_)
        packed_zero = np.packbits(e_zero, bitorder='little')
        
        if self.use_dict:
            self.dict_lut[0] = packed_zero
            self.can_correct_set.add(0)
        else:
            self.lut[0] = packed_zero
            self.can_correct[0] |= (1 << 0)
            
        if self.max_weight <= 0:
            return

        # Queue stores: (syndrome_int, error_array, weight)
        queue = deque([(0, e_zero, 0)])
        filled_count = 1
        
        pbar = None
        if self.verbose:
            pbar = tqdm(total=1 << self.m, desc="Building LUT...")
            pbar.update(1)

        # Precompute the syndrome integer for each single-qubit flip
        # to avoid matrix multiplication in the BFS loop
        H_cols = [self._syndrome_int(np.eye(self.n, dtype=np.bool_)[i]) for i in range(self.n)]

        while queue:
            if not self.use_dict and filled_count >= self.lut_size:
                break
                
            s_int, e, w = queue.popleft()

            if w >= self.max_weight:
                continue

            for i in range(self.n):
                if not e[i]:
                    new_s_int = s_int ^ H_cols[i]
                    
                    if self.use_dict:
                        if new_s_int not in self.can_correct_set:
                            new_e = e.copy()
                            new_e[i] = True
                            self.dict_lut[new_s_int] = np.packbits(new_e, bitorder='little')
                            self.can_correct_set.add(new_s_int)
                            filled_count += 1
                            if self.verbose:
                                pbar.update(1)
                            queue.append((new_s_int, new_e, w + 1))
                    else:
                        byte_idx = new_s_int >> 3
                        bit_idx = new_s_int & 7
                        if not (self.can_correct[byte_idx] & (1 << bit_idx)):
                            new_e = e.copy()
                            new_e[i] = True
                            self.lut[new_s_int] = np.packbits(new_e, bitorder='little')
                            self.can_correct[byte_idx] |= (1 << bit_idx)
                            filled_count += 1
                            if self.verbose:
                                pbar.update(1)
                            queue.append((new_s_int, new_e, w + 1))

        if self.verbose:
            pbar.close()

    def batch_decode_z(self, syndromes):
        s_ints = np.asarray(syndromes) @ self.powers_of_2
        
        if self.use_dict:
            valid_mask = np.array([s in self.can_correct_set for s in s_ints], dtype=bool)
            packed_corrections = np.zeros((len(s_ints), self.packed_len), dtype=np.uint8)
            for idx, s in enumerate(s_ints):
                if valid_mask[idx]:
                    packed_corrections[idx] = self.dict_lut[s]
        else:
            byte_idxs = s_ints >> 3
            bit_idxs = s_ints & 7
            valid_mask = (self.can_correct[byte_idxs] & (1 << bit_idxs)) != 0
            packed_corrections = self.lut[s_ints]
            
        unpacked = np.unpackbits(packed_corrections, axis=1, bitorder='little')
        corrections = unpacked[:, :self.n].astype(np.bool_)
        
        return corrections, valid_mask

    def get_mwr(self, syndrome: np.ndarray) -> np.ndarray:
        s_int = syndrome @ self.powers_of_2
        
        if self.use_dict:
            if s_int not in self.can_correct_set:
                raise RuntimeError(f"Syndrome {syndrome} not found in LutDecoder space!")
            packed = self.dict_lut[s_int]
        else:
            byte_idx = s_int >> 3
            bit_idx = s_int & 7
            if not (self.can_correct[byte_idx] & (1 << bit_idx)):
                raise RuntimeError(f"Syndrome {syndrome} not found in LutDecoder space!")
            packed = self.lut[s_int]
            
        unpacked = np.unpackbits(packed, bitorder='little')
        return unpacked[:self.n].astype(np.int8)
