import math


class Vector:
    def __init__(self, arg):
        self.x = arg[0]
        self.y = arg[1]
        self.z = arg[2] if len(arg) == 3 else 0.0

    @property
    def xy(self):
        return Vector((self.x, self.y))

    @property
    def magnitude(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    @property
    def length_squared(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    def copy(self):
        return Vector((self.x, self.y, self.z))

    def normalize(self):
        magnitude = self.magnitude

        if magnitude == 0:
            return

        self.x /= magnitude
        self.y /= magnitude
        self.z /= magnitude

    def normalized(self):
        vec = self.copy()
        vec.normalize()
        return vec

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return self.x * other.y - self.y * other.x

    def __getitem__(self, index):
        if index == 0:
            return self.x

        if index == 1:
            return self.y

        if index == 2:
            return self.z

    def __setitem__(self, index, value):
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        elif index == 2:
            self.z = value

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z

        return self

    def __add__(self, other):
        return Vector((self.x + other.x, self.y + other.y, self.z + other.z))

    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z

        return self

    def __sub__(self, other):
        return Vector((self.x - other.x, self.y - other.y, self.z - other.z))

    def __neg__(self):
        return Vector((-self.x, -self.y, -self.z))

    def __imul__(self, other):
        self.x *= other
        self.y *= other
        self.z *= other

        return self

    def __mul__(self, other):
        return Vector((self.x * other, self.y * other, self.z * other))

    def __itruediv__(self, other):
        self.x /= other
        self.y /= other
        self.z /= other

        return self

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z
