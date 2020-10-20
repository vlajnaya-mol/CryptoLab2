import numpy as np
from aes import AES


class AES_ECB(AES):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encrypt(self, stream):
        stream = stream.reshape((-1, 16))
        return np.array([self.cipher(inp) for inp in stream]).flatten()

    def decrypt(self, stream):
        stream = stream.reshape((-1, 16))
        return np.array([self.inv_cipher(inp) for inp in stream]).flatten()


class AES_CBC(AES):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_state = np.random.randint(0, 256, 16)

    def encrypt(self, stream):
        stream = stream.reshape((-1, 16))
        prev_state = self.init_state

        encoded = []
        for inp in stream:
            encoded.append(self.cipher(np.bitwise_xor(inp, prev_state)))
            prev_state = encoded[-1]
        return np.array(encoded).flatten()

    def decrypt(self, stream):
        stream = stream.reshape((-1, 16))
        prev_state = self.init_state

        decoded = []
        for inp in stream:
            decoded.append(np.bitwise_xor(self.inv_cipher(inp), prev_state))
            prev_state = inp
        return np.array(decoded).flatten()


class AES_CFB(AES):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_state = np.random.randint(0, 256, 16)
        self.n = 4

    def encrypt(self, stream):
        stream = stream.reshape((-1, self.n))

        prev_state = self.init_state

        encoded = []
        for p in stream:
            y = self.cipher(prev_state)
            encoded.append(np.bitwise_xor(p, y[:self.n]))
            prev_state = np.concatenate((prev_state[self.n:], encoded[-1]))

        return np.array(encoded).flatten()

    def decrypt(self, stream):
        stream = stream.reshape((-1, self.n))

        prev_state = self.init_state

        decoded = []
        for p in stream:
            y = self.cipher(prev_state)
            decoded.append(np.bitwise_xor(p, y[:self.n]))
            prev_state = np.concatenate((prev_state[self.n:], p))

        return np.array(decoded).flatten()


class AES_OFB(AES):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_state = np.random.randint(0, 256, 16)

    def encrypt(self, stream):
        prev_state = self.init_state
        encoded = stream.copy()
        i = 0
        while i < len(encoded):
            prev_state = self.cipher(prev_state).astype(np.uint8)
            xor_len = min(len(encoded) - i, len(prev_state))
            encoded[i:i + xor_len] ^= prev_state[:xor_len]
            i += len(prev_state)

        return encoded

    def decrypt(self, stream):
        return self.encrypt(stream)


class AES_CTR(AES):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_cnt = np.array([1] * 16)

    @staticmethod
    def count(cnt):
        for i in range(len(cnt) - 1, -1, -1):
            cnt[i] += 1
            if cnt[i] != 256:
                return cnt
            cnt[i] = 0
        return cnt

    def encrypt(self, stream):
        stream = stream.reshape((-1, 16))
        cnt = self.init_cnt.copy()
        return np.array([np.bitwise_xor(self.cipher(self.count(cnt)), inp) for inp in stream]).flatten()

    def decrypt(self, stream):
        return self.encrypt(stream)
