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

struct Sphere : public Intersectable {
    vec3 center;
    float radius;

    Sphere(const vec3 &_center, float _radius) {
        center = _center;
        radius = _radius;
    }

    Hit intersect(const Ray &ray) {
        Hit hit;
        vec3 dist = ray.start - center;
        float a = dot(ray.dir, ray.dir);
        float b = dot(dist, ray.dir) * 2.0f;
        float c = dot(dist, dist) - radius * radius;
        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;    // t1 >= t2 for sure
        float t2 = (-b - sqrt_discr) / 2.0f / a;
        if (t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = (hit.position - center) * (1.0f / radius);
        return hit;
    }
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

        for (int i = 0; i < faces.size(); i++) {
            triangles.push_back(Triangle(vertices[faces[i][0] - 1] * side_length + center,
                                         vertices[faces[i][1] - 1] * side_length + center,
                                         vertices[faces[i][2] - 1] * side_length + center,
                                         normals[faces[i][3] - 1]));
        }
    }

    Hit intersect(const Ray &ray) {
        std::vector<Hit> hits;
        for (int i = 0; i < triangles.size(); i++) {
            Hit hit = triangles[i].intersect(ray);
            if (hit.t > 0) {
                hits.push_back(hit);
            }
        }
        if (hits.size() > 0) {
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

        for (int i = 0; i < faces.size(); i++) {
            triangles.push_back(Triangle(verticles[faces[i][0] - 1] * scaling + center,
                                         verticles[faces[i][1] - 1] * scaling + center,
                                         verticles[faces[i][2] - 1] * scaling + center));
        }
    }

    Hit intersect(const Ray &ray) {
        std::vector<Hit> hits;
        for (int i = 0; i < triangles.size(); i++) {
            Hit hit = triangles[i].intersect(ray);
            if (hit.t > 0) {
                hits.push_back(hit);
            }
        }
        if (hits.size() > 0) {
            if (hits[0].t > hits[1].t) {
                return hits[1];
            } else {
                return hits[0];
            }
        } else {
            return Hit();
        }
    }

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

        for (int i = 0; i < faces.size(); i++) {
            triangles.push_back(Triangle(verticles[faces[i][0] - 1] * scaling + center,
                                         verticles[faces[i][1] - 1] * scaling + center,
                                         verticles[faces[i][2] - 1] * scaling + center));
        }
    }

    Hit intersect(const Ray &ray) {
        std::vector<Hit> hits;
        for (int i = 0; i < triangles.size(); i++) {
            Hit hit = triangles[i].intersect(ray);
            if (hit.t > 0) {
                hits.push_back(hit);
            }
        }
        if (hits.size() > 0) {
            if (hits[0].t > hits[1].t) {
                return hits[1];
            } else {
                return hits[0];
            }
        } else {
            return Hit();
        }
    }
};

struct Cone : Intersectable {
    vec3 p; // tip of the cone
    vec3 n; // unit vector in direction of increasing radius;
    float alfa; // angle between the axis and the surface
    float h; // height of the cone

    Cone(vec3 p, vec3 n, float alfa, float h) : p(p), n(n), alfa(alfa), h(h) {}


    Hit intersect(const Ray &ray) {
        Hit hit;
        vec3 s = ray.start;
        vec3 d = ray.dir;
        float a = dot(d, n) * dot(d, n) - dot(d, d) * cosf(alfa) * cosf(alfa);
        float b = 2 * dot(d, n) * dot(s - p, n) - 2 * dot(d, s - p) * cosf(alfa) * cosf(alfa);
        float c = dot(s - p, n) * dot(s - p, n) - dot(s - p, s - p) * cosf(alfa) * cosf(alfa);
        float D = b * b - 4 * a * c;
        float sqrt_discr = sqrtf(D);
        float t1 = (-b + sqrt_discr) / 2.0f / a;    // t1 >= t2 for sure
        float t2 = (-b - sqrt_discr) / 2.0f / a;
        /*hit.t = t2 > 0 ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;
        if(dot((hit.position-p) / length(hit.position-p),n) >= cosf(alfa) - 0.01f && dot(hit.position-p,n) >= 0.0f && dot(hit.position-p,n) <= h){
            hit.normal = normalize(hit.position - p - n * dot(hit.position - p, n));
            return hit;
        } else {
            hit.t = t1;
            hit.position = ray.start + ray.dir * hit.t;
            if(dot((hit.position-p) / length(hit.position-p),n) >= cosf(alfa) - 0.01f && dot(hit.position-p,n) >= 0.0f && dot(hit.position-p,n) <= h){
                hit.normal = normalize(hit.position - p - n * dot(hit.position - p, n)) * -1.0f;
                return hit;
            } else {
                return Hit();
            }
        }*/
        if(t1 < t2){
            hit.t = t1;
            hit.position = ray.start + ray.dir * hit.t;
            if(dot((hit.position-p) / length(hit.position-p),n) >= cosf(alfa) - 0.01f && dot((hit.position-p) / length(hit.position-p),n) <= cosf(alfa) + 0.01f && dot(hit.position-p,n) >= 0.0f && dot(hit.position-p,n) <= h){
                hit.normal = normalize(hit.position - p - n * dot(hit.position - p, n));
                return hit;
            } else {
                hit.t = t2;
                hit.position = ray.start + ray.dir * hit.t;
                if(dot((hit.position-p) / length(hit.position-p),n) >= cosf(alfa) - 0.01f && dot((hit.position-p) / length(hit.position-p),n) <= cosf(alfa) + 0.01f && dot(hit.position-p,n) >= 0.0f && dot(hit.position-p,n) <= h){
                    hit.normal = normalize(hit.position - p - n * dot(hit.position - p, n));
                    return hit;
                } else {
                    return Hit();
                }
            }
        }else if(t2 < t1){
            hit.t = t2;
            hit.position = ray.start + ray.dir * hit.t;
            if(dot((hit.position-p) / length(hit.position-p),n) >= cosf(alfa) - 0.01f && dot((hit.position-p) / length(hit.position-p),n) <= cosf(alfa) + 0.01f && dot(hit.position-p,n) >= 0.0f && dot(hit.position-p,n) <= h){
                hit.normal = normalize(hit.position - p - n * dot(hit.position - p, n));
                return hit;
            } else {
                hit.t = t2;
                hit.position = ray.start + ray.dir * hit.t;
                if(dot((hit.position-p) / length(hit.position-p),n) >= cosf(alfa) - 0.01f && dot((hit.position-p) / length(hit.position-p),n) <= cosf(alfa) + 0.01f && dot(hit.position-p,n) >= 0.0f && dot(hit.position-p,n) <= h){
                    hit.normal = normalize(hit.position - p - n * dot(hit.position - p, n));
                    return hit;
                } else {
                    return Hit();
                }
            }
        } else{
            return Hit();
        }


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
        return Ray(eye, dir);
    }
};

struct Light {
    vec3 direction;
    vec3 Le;

    Light(vec3 _direction, vec3 _Le) {
        direction = normalize(_direction);
        Le = _Le;
    }
};

float rnd() { return (float) rand() / RAND_MAX; }

const float epsilon = 0.0001f;

class Scene {
    std::vector<Intersectable *> objects;
    std::vector<Cone *> cones;
    std::vector<Light *> lights;
    Camera camera;
    vec3 La;
public:
    void build() {
        vec3 eye = vec3(1.0f, 1.5f, 0.0f), vup = vec3(0, 0, 1), lookat = vec3(0, 0, 0);
        float fov = 45 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        La = vec3(0.0f, 0.0f, 0.0f);
        /*vec3 lightDirection(1, 1, 1), Le(1.5, 1.5, 1.5);
        lights.push_back(new Light(lightDirection, Le));*/

        vec3 kd(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
        for (int i = 0; i < 1; i++) {
            //objects.push_back(new Sphere(vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f), rnd() * 0.1f));
            objects.push_back(new Cube(vec3(0.0f, 0.0f, 0.0f), 1));
            objects.push_back(new IcosaHedron(vec3(0.3f, 0.0f, -0.2f), 0.3f));
            objects.push_back(new DodecaHedron(vec3(-0.4f, 0.1f, -0.25f), 0.3f));
            Cone *cone = new Cone(vec3(-0.500000f,0.022291f, 0.083970f), vec3(0.0f, -1.0f, 0.0f), 0.5f, 0.5f);
            Cone *cone2 = new Cone(vec3(0.0f, 0.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f), 0.3f, 0.1f);
            Cone *cone3 = new Cone(vec3(-0.5f, 0.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f), 0.3f, 0.8f);
            cones.push_back(cone);
            cones.push_back(cone2);
            cones.push_back(cone3);
            objects.push_back(cone);
            objects.push_back(cone2);
            objects.push_back(cone3);
            //objects.push_back(new Cube(vec3(0.0f, 0.0f, 0.0f), 0.4));
            //objects.push_back(new Triangle(vec3(1.0f, 0.0f, 0.0f), vec3(1.0f, 1.0f, 1.0f), vec3(2.0f, 2.0f, 0.0f)));
        }
    }

    void refresh(int pX, int pY) {
        Hit hit = firstIntersect(camera.getRay(pX, 600 - pY));
        float shortestDistance = 1000.0f;
        Cone *closestCone;
        for (auto &cone: cones) {
            float currentLength = abs(length(hit.position - cone->p));
            if (currentLength < shortestDistance) {
                shortestDistance = currentLength;
                closestCone = cone;
            }
        }
        closestCone->p = hit.position;
        fprintf(stderr, "X: %f, Y: %f, Z: %f\n", hit.position.x, hit.position.y, hit.position.z);
        closestCone->n = hit.normal;
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

    bool shadowIntersect(Ray ray) {    // for directional lights
        for (Intersectable *object: objects) if (object->intersect(ray).t > 0) return true;
        return false;
    }

    vec3 trace(Ray ray, int depth = 0) {
        Hit hit = firstIntersect(ray);
        if (hit.t < 0) return La;
        float specularAmbient = (0.2f * (1.0f + dot(hit.normal, ray.dir * (-1.0f))));
        vec3 temp = vec3(specularAmbient, specularAmbient, specularAmbient);
        //fprintf(stderr, "Out: %f\n%f\n%f\n\n Lighted: %f\n%f\n%f\n\n", outRadiance.x, outRadiance.y, outRadiance.z,temp.x, temp.y, temp.z);
        /*for (Light *light: lights) {
            Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
            float cosTheta = dot(hit.normal, light->direction);
            if (cosTheta > 0 && !shadowIntersect(shadowRay)) {    // shadow computation
                outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
                vec3 halfway = normalize(-ray.dir + light->direction);
                float cosDelta = dot(hit.normal, halfway);
                if (cosDelta > 0)
                    outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
            }
        }*/
        return temp;
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
    printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

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