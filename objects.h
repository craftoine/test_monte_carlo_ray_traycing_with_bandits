#include "vect_3d.h"
#include "colors.h"
struct Material {
    float reflection_coefficient;
    float diffuse_coefficient;
    Material(float reflection = 0.0f, float diffuse = 0.0f) : reflection_coefficient(reflection), diffuse_coefficient(diffuse) {}
};

struct SceneObject {
    Vector3D position;
    float radius;
    Color color;
    Material material;
    float emission;
    SceneObject(Vector3D pos, float rad, Color col, Material mat, float emit) : position(pos), radius(rad), color(col), material(mat), emission(emit) {}
};