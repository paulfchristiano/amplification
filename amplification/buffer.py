import numpy as np
from collections import defaultdict
import threading

class Buffer:
    def __init__(self, capacity, shapes, validation_fraction=0):
        self.capacity = capacity
        self.used = 0
        self.index = 0
        self.shapes = shapes
        self.buffer = {name:np.ones([capacity] + list(shape), dtype=np.int32) * 7 for name, shape in shapes.items()}
        self.validation_fraction = validation_fraction
        self.validate = (validation_fraction > 0)
        self.total_data = 0
        self.lock = threading.RLock()
        if self.validate:
            self.validation_owed = 0
            self._validation_buffer = Buffer(int(capacity * validation_fraction), shapes)
            #no one should acquire _validation_buffer.lock without acquiring self.lock first

    def keys(self):
        return list(self.buffer)

    def extend(self, stuff, batch=True, extendible={}):
        with self.lock:
            if not batch: stuff = {n: np.asarray([x]) for n, x in stuff.items()}

            assert set(stuff) == set(self.buffer)
            nbatches = [x.shape[0] for x in stuff.values()]
            nbatch = nbatches[0]
            assert all(n == nbatch for n in nbatches)

            for n in extendible:
                for index in extendible[n]:
                    newsize = stuff[n].shape[index]
                    self.grow(n, index, newsize)

            if self.validate:
                self.validation_owed += self.validation_fraction * nbatch
                validation_cut = min(int(self.validation_owed), nbatch)
                self.validation_owed -= validation_cut
                validation_stuff = {v:k[:validation_cut] for v, k in stuff.items()}
                self._validation_buffer.extend(validation_stuff)
                stuff = {v:k[validation_cut:] for v, k in stuff.items()}
                nbatch -= validation_cut

            self.total_data += nbatch

            # if they're trying to insert more stuff than would fit all at once, only insert the last bit
            item_size = list(stuff.values())[0].shape[0]
            if item_size > self.capacity:
                stuff = {name:val[-self.capacity:] for name, val in stuff.items()}

            if self.index + item_size <= self.capacity:
                for n in self.buffer:
                    self.buffer[n][self.index : self.index + item_size] = stuff[n]
                self.index += item_size
                if self.index > self.used:
                    self.used = self.index
            else:
                space_at_end = self.capacity - self.index
                for n in self.buffer:
                    self.buffer[n][self.index:] = stuff[n][:space_at_end]
                    self.buffer[n][:item_size - space_at_end] = stuff[n][space_at_end:]

                self.index = (item_size + self.used) % self.capacity
                self.used = self.capacity

    def grow(self, k, index, newsize):
        with self.lock:
            old = self.buffer[k]
            oldsize = old.shape[index]
            if newsize != oldsize:
                assert newsize > oldsize
                to_set = [slice(0, s) for s in old.shape]
                shape = list(old.shape)
                shape[index] = newsize
                self.buffer[k] = np.zeros(shape, dtype=np.int32)
                self.buffer[k][to_set] = old
            if self.validate:
                self._validation_buffer.grow(k, index, newsize)


    def sample(self, n, validation=False):
        with self.lock:
            if validation:
                return self._validation_buffer.sample(n)

            if n > self.used:
                raise IndexError("You're requesting more items than exist in the buffer")

            indexes = np.random.randint(0, self.used, size=n)

            return {n:self.buffer[n][indexes] for n in self.buffer}

    def has(self, n):
        return n <= self.used

if __name__ == '__main__':
    b = Buffer(5, [[0, 2], [0, 2]])
    b.extend([np.ones((3, 2, 2)) * 5, np.ones((3, 3, 2)) * 11])
    print(b.buffer)

    # b.extend([np.asarray([[4, 5], [6, 7], [8, 9]])])
    # print(b.buffer)
    print("Sampling")
    print(b.sample(2))
    # print(b.sample(3))
    # print(b.sample(3))
