import numpy as np


class RC4:
    def __init__(self, Key):
        self.Key = Key.copy()
        self.S_init = self.__ksa(self.Key)
    
    @staticmethod
    def __ksa(Key):
        if len(Key) < 256:
            Key = np.tile(Key, round(256 / len(Key) + 0.5))
        S = np.arange(0, 256, dtype=np.uint8)
        j = np.uint8(0)
        for i in range(256):
            j += S[i] + Key[i]
            S[i], S[j] = S[j], S[i]
        return S
    
    def __prga(self):
        S = self.S_init.copy()
        i, j = np.uint8(0), np.uint8(0)
        one = np.uint8(1)
        while True:
            i += one
            j += S[i]
            S[i], S[j] = S[j], S[i]
            t = S[i] + S[j]
            yield S[t]

    def encode(self, stream_bytes):
        gen = self.__prga()
        return np.array(list(map(lambda x: np.bitwise_xor(x, next(gen)), stream_bytes)))
    
    def decode(self, encoded_bytes):
        return self.encode(encoded_bytes)