struct Color {
    float r, g, b;
    Color(float red = 0.0f, float green = 0.0f, float blue = 0.0f) : r(red), g(green), b(blue) {}
    
    Color operator*(float scallar) const {
        return Color(r * scallar, g * scallar, b * scallar);
    }
    Color operator*=(float scallar) {
        r *= scallar;
        g *= scallar;
        b *= scallar;
        return *this;
    }
    Color operator+(const Color& other) const {
        return Color(r + other.r, g + other.g, b + other.b);
    }
    Color& operator+=(const Color& other) {
        r += other.r;
        g += other.g;
        b += other.b;
        return *this;
    }
    Color operator*(const Color& other) const {
        return Color(r * other.r, g * other.g, b * other.b);
    }
    Color& operator*=(const Color& other) {
        r *= other.r;
        g *= other.g;
        b *= other.b;
        return *this;
    }
    /*define color>color and < for array definition to work*/
    bool operator>(const Color& other) const {
        return r > other.r && g > other.g && b > other.b;
    }
    bool operator<(const Color& other) const {
        return r < other.r && g < other.g && b < other.b;
    }
    /*define color/color for array definition to work*/
    Color operator/(const Color& other) const {
        return Color(r / other.r, g / other.g, b / other.b);
    }
    /*define color/scallar for array definition to work*/
    Color operator/(float scallar) const {
        return Color(r / scallar, g / scallar, b / scallar);
    }
    /*define color-color for array definition to work*/
    Color operator-(const Color& other) const {
        return Color(r - other.r, g - other.g, b - other.b);
    }
    /*define color-scallar for array definition to work*/
    Color operator-(float scallar) const {
        return Color(r - scallar, g - scallar, b - scallar);
    }
    /*overload the << operator*/
    friend std::ostream& operator<<(std::ostream& os, const Color& color) {
        os << "(" << color.r << ", " << color.g << ", " << color.b << ")";
        return os;
    }
};