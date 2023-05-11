//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
#include "framework.h"

struct Hit {
    float t;
    vec3 position, normal;

    Hit() { t = -1; }
};

struct Ray {
    vec3 start, dir;

    Ray(vec3 _start, vec3 _dir) {
        start = _start;
        dir = normalize(_dir);
    }
};

class Intersectable {
public:
    virtual Hit intersect(const Ray &ray) = 0;
};

// Triangle intersection based on 12. page of the ray-tracing pdf.
struct Triangle : public Intersectable {
    vec3 r1, r2, r3;
    vec3 n;

    Triangle(const vec3 &_r1, const vec3 &_r2, const vec3 &_r3) {
        r1 = _r1;
        r2 = _r2;
        r3 = _r3;
        n = normalize(cross(r2 - r1, r3 - r1));
    }

    Triangle(const vec3 &_r1, const vec3 &_r2, const vec3 &_r3, const vec3 &_n) {
        r1 = _r1;
        r2 = _r2;
        r3 = _r3;
        n = _n;
    }

    Hit intersect(const Ray &ray) {
        Hit hit;
        float t = dot(r1 - ray.start, n) / dot(ray.dir, n);
        if (t <= 0) return hit;
        vec3 p = ray.start + ray.dir * t;
        if (dot(cross(r2 - r1, p - r1), n) > 0 &&
            dot(cross(r3 - r2, p - r2), n) > 0 &&
            dot(cross(r1 - r3, p - r3), n) > 0) {
            hit.t = t;
            hit.position = p;
            hit.normal = n;
        }
        return hit;
    }
};

struct Cube : public Intersectable {
    std::vector<Triangle> triangles;
    vec3 center;
    float side_length;

    Cube(const vec3 &center, const float &side_length)
            : center(center), side_length(side_length) {
        std::vector<vec3> vertices = {
                vec3(-0.5f, -0.5f, -0.5f),
                vec3(-0.5f, -0.5f, 0.5f),
                vec3(-0.5f, 0.5f, -0.5f),
                vec3(-0.5f, 0.5f, 0.5f),
                vec3(0.5f, -0.5f, -0.5f),
                vec3(0.5f, -0.5f, 0.5f),
                vec3(0.5f, 0.5f, -0.5f),
                vec3(0.5f, 0.5f, 0.5f)
        };

        std::vector<std::vector<int>> faces = {
                // the first 3 numbers are the indices of the vertices of the triangle, the last is the normal vectors.
                {1, 7, 5, 2},
                {1, 3, 7, 2},
                {1, 4, 3, 6},
                {1, 2, 4, 6},
                {3, 8, 7, 3},
                {3, 4, 8, 3},
                {5, 7, 8, 5},
                {5, 8, 6, 5},
                {1, 5, 6, 4},
                {1, 6, 2, 4},
                {2, 6, 8, 1},
                {2, 8, 4, 1}
        };

        std::vector<vec3> normals = {
                vec3(0.0f, 0.0f, 1.0f),
                vec3(0.0f, 0.0f, -1.0f),
                vec3(0.0f, 1.0f, 0.0f),
                vec3(0.0f, -1.0f, 0.0f),
                vec3(1.0f, 0.0f, 0.0f),
                vec3(-1.0f, 0.0f, 0.0f)
        };

        for (auto & face : faces) {
            triangles.emplace_back(vertices[face[0] - 1] * side_length + center,
                                         vertices[face[1] - 1] * side_length + center,
                                         vertices[face[2] - 1] * side_length + center,
                                         normals[face[3] - 1]);
        }
    }

    Hit intersect(const Ray &ray) override {
        std::vector<Hit> hits;
        for (auto & triangle : triangles) {
            Hit hit = triangle.intersect(ray);
            if (hit.t > 0) {
                hits.push_back(hit);
            }
        }
        if (!hits.empty()) {
            if (hits[0].t < hits[1].t) {
                return hits[1];
            } else {
                return hits[0];
            }
        } else {
            return Hit();
        }
    }
};

struct IcosaHedron : public Intersectable {
    vec3 center;
    float scaling;
    std::vector<Triangle> triangles;

    IcosaHedron(const vec3 &center, const float &scaling) : center(center), scaling(scaling) {
        std::vector<vec3> verticles = {
                vec3(0, -0.525731, 0.850651),
                vec3(0.850651, 0, 0.525731),
                vec3(0.850651, 0, -0.525731),
                vec3(-0.850651, 0, -0.525731),
                vec3(-0.850651, 0, 0.525731),
                vec3(-0.525731, 0.850651, 0),
                vec3(0.525731, 0.850651, 0),
                vec3(0.525731, -0.850651, 0),
                vec3(-0.525731, -0.850651, 0),
                vec3(0, -0.525731, -0.850651),
                vec3(0, 0.525731, -0.850651),
                vec3(0, 0.525731, 0.850651)
        };
        std::vector<std::vector<int>> faces = {
                {2,  3,  7},
                {2,  8,  3},
                {4,  5,  6},
                {5,  4,  9},
                {7,  6,  12},
                {6,  7,  11},
                {10, 11, 3},
                {11, 10, 4},
                {8,  9,  10},
                {9,  8,  1},
                {12, 1,  2},
                {1,  12, 5},
                {7,  3,  11},
                {2,  7,  12},
                {4,  6,  11},
                {6,  5,  12},
                {3,  8,  10},
                {8,  2,  1},
                {4,  10, 9},
                {5,  9,  1}
        };

        for (auto & face : faces) {
            triangles.emplace_back(verticles[face[0] - 1] * scaling + center,
                                         verticles[face[1] - 1] * scaling + center,
                                         verticles[face[2] - 1] * scaling + center);
        }
    }

    Hit intersect(const Ray &ray) {
        std::vector<Hit> hits;
        for (auto & triangle : triangles) {
            Hit hit = triangle.intersect(ray);
            if (hit.t > 0) {
                hits.push_back(hit);
            }
        }
        if (!hits.empty()) {
            if (hits[0].t > hits[1].t) {
                return hits[1];
            } else {
                return hits[0];
            }
        } else {
            return {};
        }
    }

};

struct Light {
    vec3 position;
    vec3 Le;

    Light(vec3 position, vec3 Le) : position(position), Le(Le) {}
};

struct DodecaHedron : public Intersectable {
    std::vector<Triangle> triangles;

    DodecaHedron(vec3 center, float scaling) {
        std::vector<vec3> verticles = {
                vec3(-0.57735, -0.57735, 0.57735),
                vec3(0.934172, 0.356822, 0),
                vec3(0.934172, -0.356822, 0),
                vec3(-0.934172, 0.356822, 0),
                vec3(-0.934172, -0.356822, 0),
                vec3(0, 0.934172, 0.356822),
                vec3(0, 0.934172, -0.356822),
                vec3(0.356822, 0, -0.934172),
                vec3(-0.356822, 0, -0.934172),
                vec3(0, -0.934172, -0.356822),
                vec3(0, -0.934172, 0.356822),
                vec3(0.356822, 0, 0.934172),
                vec3(-0.356822, 0, 0.934172),
                vec3(0.57735, 0.57735, -0.57735),
                vec3(0.57735, 0.57735, 0.57735),
                vec3(-0.57735, 0.57735, -0.57735),
                vec3(-0.57735, 0.57735, 0.57735),
                vec3(0.57735, -0.57735, -0.57735),
                vec3(0.57735, -0.57735, 0.57735),
                vec3(-0.57735, -0.57735, -0.57735)
        };
        std::vector<std::vector<int>> faces = {
                {19, 3,  2},
                {12, 19, 2},
                {15, 12, 2},
                {8,  14, 2},
                {18, 8,  2},
                {3,  18, 2},
                {20, 5,  4},
                {9,  20, 4},
                {16, 9,  4},
                {13, 17, 4},
                {1,  13, 4},
                {5,  1,  4},
                {7,  16, 4},
                {6,  7,  4},
                {17, 6,  4},
                {6,  15, 2},
                {7,  6,  2},
                {14, 7,  2},
                {10, 18, 3},
                {11, 10, 3},
                {19, 11, 3},
                {11, 1,  5},
                {10, 11, 5},
                {20, 10, 5},
                {20, 9,  8},
                {10, 20, 8},
                {18, 10, 8},
                {9,  16, 7},
                {8,  9,  7},
                {14, 8,  7},
                {12, 15, 6},
                {13, 12, 6},
                {17, 13, 6},
                {13, 1,  11},
                {12, 13, 11},
                {19, 12, 11}};

        for (auto & face : faces) {
            triangles.emplace_back(verticles[face[0] - 1] * scaling + center,
                                         verticles[face[1] - 1] * scaling + center,
                                         verticles[face[2] - 1] * scaling + center);
        }
    }

    Hit intersect(const Ray &ray) {
        std::vector<Hit> hits;
        for (auto & triangle : triangles) {
            Hit hit = triangle.intersect(ray);
            if (hit.t > 0) {
                hits.push_back(hit);
            }
        }
        if (!hits.empty()) {
            if (hits[0].t > hits[1].t) {
                return hits[1];
            } else {
                return hits[0];
            }
        } else {
            return {};
        }
    }
};

const float epsilon = 0.0001f;

struct Cone : Intersectable {
    Light *light;
    vec3 p; // tip of the cone
    vec3 n; // unit vector in direction of increasing radius;
    float alfa; // angle between the axis and the surface
    float h; // height of the cone

    Cone(vec3 p, vec3 n, float alfa, float h, Light *light) : light(light), p(p), n(n), alfa(alfa), h(h) {}

    Hit intersect(const Ray& ray) {
        Hit hit;
        vec3 s = ray.start;
        vec3 d = ray.dir;
        float a = dot(d, n) * dot(d, n) - dot(d, d) * cosf(alfa) * cosf(alfa);
        float b = 2 * dot(d, n) * dot(s - p, n) - 2 * dot(d, s - p) * cosf(alfa) * cosf(alfa);
        float c = dot(s - p, n) * dot(s - p, n) - dot(s - p, s - p) * cosf(alfa) * cosf(alfa);
        float D = b * b - 4 * a * c;
        float sqrt_discr = sqrtf(D);
        float t1 = (-b + sqrt_discr) / (2.0f * a);
        float t2 = (-b - sqrt_discr) / (2.0f * a);

        bool t1Valid = false;
        bool t2Valid = false;

        if (t1 < t2) {
            hit.t = t1;
            hit.position = ray.start + ray.dir * hit.t;
            t1Valid = checkIntersection(hit, p, n, alfa, h);

            if (!t1Valid) {
                hit.t = t2;
                hit.position = ray.start + ray.dir * hit.t;
                t2Valid = checkIntersection(hit, p, n, alfa, h);
            }
        } else if (t2 < t1) {
            hit.t = t2;
            hit.position = ray.start + ray.dir * hit.t;
            t2Valid = checkIntersection(hit, p, n, alfa, h);

            if (!t2Valid) {
                hit.t = t1;
                hit.position = ray.start + ray.dir * hit.t;
                t1Valid = checkIntersection(hit, p, n, alfa, h);
            }
        }

        if (t1Valid || t2Valid) {
            hit.normal = normalize(hit.position - p - n * dot(hit.position - p, n));
            if (t2Valid) {
                hit.normal = hit.normal * -1.0f;
            }
            return hit;
        } else {
            return {};
        }
    }

    bool checkIntersection(const Hit& hit, const vec3& p, const vec3& n, float alfa, float h) {
        vec3 positionDiff = hit.position - p;
        float positionLength = length(positionDiff);
        float dotProduct = dot(positionDiff / positionLength, n);

        return (dotProduct >= cosf(alfa) - epsilon * 40.0f && dotProduct <= cosf(alfa) + epsilon * 40.0f &&
                dot(hit.position - p, n) >= 0.0f && dot(hit.position - p, n) <= h);
    }


};

class Camera {
    vec3 eye, lookat, right, up;
public:
    void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
        eye = _eye;
        lookat = _lookat;
        vec3 w = eye - lookat;
        float focus = length(w);
        right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
        up = normalize(cross(w, right)) * focus * tanf(fov / 2);
    }

    Ray getRay(int X, int Y) {
        vec3 dir =
                lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) -
                eye;
        return {eye, dir};
    }
};

float rnd() { return (float) rand() / RAND_MAX; }


class Scene {
    std::vector<Intersectable *> objects;
    std::vector<Cone *> cones;
    Camera camera;
    vec3 La;
public:
    void build() {
        vec3 eye = vec3(1.0f, 1.5f, 0.0f), vup = vec3(0, 0, 1), lookat = vec3(0, 0, 0);
        float fov = 45 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        La = vec3(0.0f, 0.0f, 0.0f);

        // The outer cube.
        objects.push_back(new Cube(vec3(0.0f, 0.0f, 0.0f), 1));

        // The Icosahedron on the left hand side.
        objects.push_back(new IcosaHedron(vec3(0.3f, 0.0f, -0.2f), 0.3f));

        // The Dodecahedron on the right hand side.
        objects.push_back(new DodecaHedron(vec3(-0.4f, 0.1f, -0.25f), 0.3f));

        // Getting some random coordinates for the cones.
        Hit hRedCone = firstIntersect(camera.getRay(450, 500));
        Hit hGreenCone = firstIntersect(camera.getRay(234, 160));
        Hit hBlueCone = firstIntersect(camera.getRay(123, 345));

        // Creating the lights.
        auto *redLight = new Light(hRedCone.position + hRedCone.normal * epsilon * 50, vec3(0.2f, 0.0f, 0.0f));
        auto *greenLight = new Light(hGreenCone.position + hGreenCone.normal * epsilon * 50, vec3(0.0f, 0.2f, 0.0f));
        auto *blueLight = new Light(hBlueCone.position + hBlueCone.normal * epsilon * 50, vec3(0.0f, 0.0f, 0.2f));

        // Creating the 3 cone (Listening device), storing them separately,
        // to use them as lights, lights are connected to the cones.
        Cone *redCone = new Cone(hRedCone.position - hRedCone.normal * 40 *epsilon, hRedCone.normal, 0.4, 0.1f, redLight);
        Cone *greenCone = new Cone(hGreenCone.position - hGreenCone.normal * 40* epsilon, hGreenCone.normal, 0.4f, 0.1f, greenLight);
        Cone *blueCone = new Cone(hBlueCone.position - hBlueCone.normal * 40 *epsilon, hBlueCone.normal, 0.4f, 0.1f, blueLight);

        objects.push_back(redCone);
        objects.push_back(greenCone);
        objects.push_back(blueCone);

        cones.push_back(redCone);
        cones.push_back(greenCone);
        cones.push_back(blueCone);
    }

    void refresh(int pX, int pY) {
        Hit hit = firstIntersect(camera.getRay(pX, 600 - pY));
        float shortestDistance = 1000.0f;
        Cone *closestCone;
        for (auto cone: cones) {
            fprintf(stderr, "cone: %f %f %f\n", cone->p.x, cone->p.y, cone->p.z);
            float currentLength = abs(length(hit.position - cone->p));
            if (currentLength < shortestDistance) {
                shortestDistance = currentLength;
                closestCone = cone;
            }
        }
        closestCone->p = hit.position - hit.normal * epsilon * 40;
        closestCone->n = hit.normal;
        closestCone->light->position = hit.position + closestCone->n * epsilon * 50;
    }

    void render(std::vector<vec4> &image) {
        for (int Y = 0; Y < windowHeight; Y++) {
                #pragma omp parallel for
            for (int X = 0; X < windowWidth; X++) {
                vec3 color = trace(camera.getRay(X, Y));
                image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
            }
        }
    }

    Hit firstIntersect(Ray ray) {
        Hit bestHit;
        for (Intersectable *object: objects) {
            Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) bestHit = hit;
        }
        if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
        return bestHit;
    }

    vec3 trace(Ray ray, int depth = 0) {
        Hit hit = firstIntersect(ray);
        if (hit.t < 0) return La;
        float ambientFactor = (0.2f * (1.0f + dot(hit.normal, ray.dir * (-1.0f))));
        vec3 specularAmbient = vec3(ambientFactor, ambientFactor, ambientFactor);
        for (Cone *cone: cones) {
            Ray rayToLight = Ray(hit.position + hit.normal * epsilon / 2.0f,
                                 normalize(cone->light->position - hit.position));
            Hit hitToLight = firstIntersect(rayToLight);
            if (length(hitToLight.position - hit.position) > length(cone->light->position - hit.position)) {
                specularAmbient = specularAmbient + cone->light->Le * (1.0f / hitToLight.t);
            }

        }
        return specularAmbient;
    }
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
    unsigned int vao;    // vertex array object id and texture id
    Texture texture;
public:
    FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4> &image)
            : texture(windowWidth, windowHeight, image) {
        glGenVertexArrays(1, &vao);    // create 1 vertex array object
        glBindVertexArray(vao);        // make it active

        unsigned int vbo;        // vertex buffer objects
        glGenBuffers(1, &vbo);    // Generate 1 vertex buffer objects

        // vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        float vertexCoords[] = {-1, -1, 1, -1, 1, 1, -1, 1};    // two triangles forming a quad
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords,
                     GL_STATIC_DRAW);       // copy to that part of the memory which is not modified
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
    }

    void Draw() {
        glBindVertexArray(vao);    // make the vao and its vbos active playing the role of the data source
        gpuProgram.setUniform(texture, "textureUnit");
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);    // draw two triangles forming a quad
    }
};

FullScreenTexturedQuad *fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();

    std::vector<vec4> image(windowWidth * windowHeight);
    long timeStart = glutGet(GLUT_ELAPSED_TIME);
    scene.render(image);
    long timeEnd = glutGet(GLUT_ELAPSED_TIME);
    printf("Rendering time: %ld milliseconds\n", (timeEnd - timeStart));

    // copy image to GPU as a texture
    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

    // create program for the GPU
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
    fullScreenTexturedQuad->Draw();
    glutSwapBuffers();                                    // exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
    if (state == GLUT_DOWN) {
        if (button == GLUT_LEFT_BUTTON) {
            scene.refresh(pX, pY);

            std::vector<vec4> image(windowWidth * windowHeight);
            scene.render(image);
            delete fullScreenTexturedQuad;
            fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
            fullScreenTexturedQuad->Draw();
            glutSwapBuffers();
        }
    }
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}	