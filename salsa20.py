import numpy as np


class Salsa20:
    def __init__(self, Key, rounds=20):
        if Key.dtype == np.uint8:
            Key = self.bytes_to_32bits(Key)
        self.Key = Key
        self.rounds = rounds
    
    @staticmethod
    def __qr(a, b, c, d):
        def rot(a, n):
            return (a << a) | (a >> (np.uint32(32) - np.uint32(n)))

        e = a ^ rot(d + c, 18)
        f = b ^ rot(a + d, 7)
        g = c ^ rot(b + a, 9)
        h = d ^ rot(c + b, 13)
        return (e, f, g, h)
    
    def __salsa20(self, stream_16_32bit, pos_32bit):
        nonce = self.__get_nonce(pos_32bit)
        S = np.array([[0x61707865, *self.Key[:3]],
                      [self.Key[3], 0x3320646e, *nonce],
                      [*pos_32bit, 0x79622d32, self.Key[4]],
                      [*self.Key[5:], 0x6b206574]], dtype=np.uint32)

        for r in range(1, self.rounds + 1):
            if r % 2 == 1:
                S[:, 0] = self.__qr(*S[[0, 1, 2, 3], 0])
                S[:, 1] = self.__qr(*S[[1, 2, 3, 0], 1])
                S[:, 2] = self.__qr(*S[[2, 3, 0, 1], 2])
                S[:, 3] = self.__qr(*S[[3, 0, 1, 2], 3])
            else:
                S[0, :] = self.__qr(*S[0, [0, 1, 2, 3]])
                S[1, :] = self.__qr(*S[1, [1, 2, 3, 0]])
                S[2, :] = self.__qr(*S[2, [2, 3, 0, 1]])
                S[3, :] = self.__qr(*S[3, [3, 0, 1, 2]])

        return np.bitwise_xor(stream_16_32bit, S.flatten())

    @staticmethod
    def bytes_to_32bits(stream_bytes):
        x = np.pad(stream_bytes, (0, round(len(stream_bytes) / 4 + 0.5) * 4 - len(stream_bytes)), constant_values=np.uint8(0))
        x = x.reshape((-1, 4)).astype(np.uint32)
        x[:, 0] = x[:, 0] << np.uint32(24)
        x[:, 1] = x[:, 1] << np.uint32(16)
        x[:, 2] = x[:, 2] << np.uint32(8)
        return x.sum(axis=1).astype(np.uint32)

    @staticmethod
    def bytes_from_32bits(stream_32bits):
        dt = np.dtype((np.uint32, {'f0':(np.uint8,0),'f1':(np.uint8,1),'f2':(np.uint8,2), 'f3':(np.uint8,3)}))
        x = stream_32bits.view(dtype=dt)
        x = np.array([x["f3"], x["f2"], x["f1"], x["f0"]], dtype=np.uint8)
        x = x.T.flatten()
        return x
    
    @staticmethod
    def int64_to_ints32(int64):
        return np.array([int64 >> np.uint64(32), int64 & np.uint64(0xFFFFFFFF)], dtype=np.uint32)
    
    @staticmethod
    def __get_nonce(arr_32bits):
        hashed_stream = hash((np.uint64(arr_32bits[0]) << np.uint64(32)) + np.uint64(arr_32bits[1]))
        return Salsa20.int64_to_ints32(np.uint64(hashed_stream))

    def encode(self, stream):
        if stream.dtype == np.uint8:
            stream = self.bytes_to_32bits(stream)
            
        assert len(stream) % 16 == 0
        
        encoded = np.array([self.__salsa20(stream[i:i + 16], 
                                           self.int64_to_ints32(np.uint64(i // 16)))
                            for i in range(0, len(stream), 16)], dtype=np.uint32).flatten()
        
        return self.bytes_from_32bits(encoded)
    
    def decode(self, stream):
        return self.encode(stream)
        