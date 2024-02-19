struct Vector3D {
    float x, y, z;

    Vector3D(float x_val = 0.0f, float y_val = 0.0f, float z_val = 0.0f) : x(x_val), y(y_val), z(z_val) {}

    Vector3D operator+(const Vector3D& other) const {
        return Vector3D(x + other.x, y + other.y, z + other.z);
    }
    Vector3D& operator+=(const Vector3D& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
    Vector3D operator-(const Vector3D& other) const {
        return Vector3D(x - other.x, y - other.y, z - other.z);
    }

    Vector3D operator*(float scallar) const {
        return Vector3D(x * scallar, y * scallar, z * scallar);
    }
    Vector3D& operator*=(float scallar) {
        x *= scallar;
        y *= scallar;
        z *= scallar;
        return *this;
    }
    Vector3D operator*(const Vector3D& other) const {
        return Vector3D(x * other.x, y * other.y, z * other.z);
    }
    Vector3D& operator*=(const Vector3D& other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        return *this;
    }
    float dot(const Vector3D& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    Vector3D cross(const Vector3D& other) const {
        return Vector3D(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
    }

    float magnitude() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    Vector3D normalize() const {
        float mag = magnitude();
        return Vector3D(x / mag, y / mag, z / mag);
    }
    /*overload the << operator*/
    friend std::ostream& operator<<(std::ostream& os, const Vector3D& vector) {
        os << "Vector3D(" << vector.x << ", " << vector.y << ", " << vector.z << ")";
        return os;
    }
};