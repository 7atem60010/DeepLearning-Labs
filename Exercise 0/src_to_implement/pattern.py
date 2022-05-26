import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        if resolution % (2 * tile_size) != 0:
            raise "Dimension Error"
        tile_size_square = (tile_size, tile_size)
        tile0 = np.concatenate((np.ones(tile_size_square), np.zeros(tile_size_square)), axis=0)
        tile1 = np.concatenate((np.zeros(tile_size_square), np.ones(tile_size_square)), axis=0)
        tile = np.concatenate((tile1, tile0), axis=1)
        self.output = np.tile(tile, (resolution // (2 * tile_size), resolution // (2 * tile_size)))
        return

    def draw(self):
        return self.output.copy()
        pass

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()
        pass


def draw(self):
    return self.output.copy()
    pass


def show(self):
    plt.imshow(self.output, cmap='gray')
    plt.show()
    pass


class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = np.zeros((self.resolution, self.resolution), dtype=int)
        p2, p1 = position
        x_points = np.linspace(p1 - radius, p1 + radius, radius * 2 + 1, dtype=int)
        y_points = np.linspace(p2 - radius, p2 + radius, radius * 2 + 1, dtype=int)
        x, y = np.meshgrid(x_points, y_points)
        xx, yy = x.reshape((1, -1)), y.reshape((1, -1))
        inside = np.less_equal((xx - p1) ** 2 + (yy - p2) ** 2, radius ** 2).astype(int)
        np.set_printoptions(threshold=np.inf)
        self.output[xx, yy] = inside

    def draw(self):
        return self.output.copy()
        pass

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = np.zeros((self.resolution, self.resolution, 3))
        dim = np.linspace(0, resolution - 1, resolution) / resolution
        self.output[:, :, 0] = dim  # red
        self.output[:, :, 2] = dim[::-1]  # blue
        self.output[:, :, 1] = dim  # green
        self.output[:, :, 1] = self.output[:, :, 1].T
        return

    def draw(self):
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()
        pass

