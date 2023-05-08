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

struct Cube : public Intersectable {
    vec3 center;
    float side_dimension; // length of the side of the cube


    Cube(const vec3 &_center, float _side_dimension) {
        center = _center;
        side_dimension = _side_dimension;
    }

    // should result negative t parameter if no intersect is found
    Hit intersect(const Ray &ray) {
        Hit hit;
        std::pair<vec3, float> intersections[6];
        vec3 normals[6];
        int numIntersections = 0;
        vec3 centerOfTop = center + vec3(0, side_dimension / 2, 0);
        vec3 normalOfTop = vec3(0, 1, 0);
        std::pair<vec3, float> Top = intersectionOfStraightLineAndPlane(ray.start, ray.dir, centerOfTop, normalOfTop);
        if (Top.first.y < centerOfTop.y + side_dimension / 2 && Top.first.y > centerOfTop.y - side_dimension / 2 &&
            Top.first.x < centerOfTop.x + side_dimension / 2 && Top.first.x > centerOfTop.x - side_dimension / 2 &&
            Top.first.z < centerOfTop.z + side_dimension / 2 && Top.first.z > centerOfTop.z - side_dimension / 2) {
            intersections[numIntersections] = Top;
            normals[numIntersections++] = normalOfTop;
        }

        vec3 centerOfBottom = center - vec3(0, side_dimension / 2, 0);
        vec3 normalOfBottom = vec3(0, -1, 0);
        std::pair<vec3, float> Bottom = intersectionOfStraightLineAndPlane(ray.start, ray.dir, centerOfBottom,
                                                                           normalOfBottom);
        if (Bottom.first.y < centerOfBottom.y + side_dimension / 2 &&
            Bottom.first.y > centerOfBottom.y - side_dimension / 2 &&
            Bottom.first.x < centerOfBottom.x + side_dimension / 2 &&
            Bottom.first.x > centerOfBottom.x - side_dimension / 2 &&
            Bottom.first.z < centerOfBottom.z + side_dimension / 2 &&
            Bottom.first.z > centerOfBottom.z - side_dimension / 2) {
            intersections[numIntersections] = Bottom;
            normals[numIntersections++] = normalOfBottom;
        }

        vec3 centerOfLeft = center - vec3(side_dimension / 2, 0, 0);
        vec3 normalOfLeft = vec3(-1, 0, 0);
        std::pair<vec3, float> Left = intersectionOfStraightLineAndPlane(ray.start, ray.dir, centerOfLeft,
                                                                         normalOfLeft);
        if (Left.first.y < centerOfLeft.y + side_dimension / 2 && Left.first.y > centerOfLeft.y - side_dimension / 2 &&
            Left.first.x < centerOfLeft.x + side_dimension / 2 && Left.first.x > centerOfLeft.x - side_dimension / 2 &&
            Left.first.z < centerOfLeft.z + side_dimension / 2 && Left.first.z > centerOfLeft.z - side_dimension / 2) {
            intersections[numIntersections] = Left;
            normals[numIntersections++] = normalOfLeft;
        }

        vec3 centerOfRight = center + vec3(side_dimension / 2, 0, 0);
        vec3 normalOfRight = vec3(1, 0, 0);
        std::pair<vec3, float> Right = intersectionOfStraightLineAndPlane(ray.start, ray.dir, centerOfRight,
                                                                          normalOfRight);
        if (Right.first.y < centerOfRight.y + side_dimension / 2 &&
            Right.first.y > centerOfRight.y - side_dimension / 2 &&
            Right.first.x < centerOfRight.x + side_dimension / 2 &&
            Right.first.x > centerOfRight.x - side_dimension / 2 &&
            Right.first.z < centerOfRight.z + side_dimension / 2 &&
            Right.first.z > centerOfRight.z - side_dimension / 2) {
            intersections[numIntersections] = Right;
            normals[numIntersections++] = normalOfRight;
        }

        vec3 centerOfFront = center + vec3(0, 0, side_dimension / 2);
        vec3 normalOfFront = vec3(0, 0, 1);
        std::pair<vec3, float> Front = intersectionOfStraightLineAndPlane(ray.start, ray.dir, centerOfFront,
                                                                          normalOfFront);
        if (Front.first.y < centerOfFront.y + side_dimension / 2 &&
            Front.first.y > centerOfFront.y - side_dimension / 2 &&
            Front.first.x < centerOfFront.x + side_dimension / 2 &&
            Front.first.x > centerOfFront.x - side_dimension / 2 &&
            Front.first.z < centerOfFront.z + side_dimension / 2 &&
            Front.first.z > centerOfFront.z - side_dimension / 2) {
            intersections[numIntersections] = Front;
            normals[numIntersections++] = normalOfFront;
        }

        vec3 centerOfBack = center - vec3(0, 0, side_dimension / 2);
        vec3 normalOfBack = vec3(0, 0, -1);
        std::pair<vec3, float> Back = intersectionOfStraightLineAndPlane(ray.start, ray.dir, centerOfBack,
                                                                         normalOfBack);
        if (Back.first.y < centerOfBack.y + side_dimension / 2 && Back.first.y > centerOfBack.y - side_dimension / 2 &&
            Back.first.x < centerOfBack.x + side_dimension / 2 && Back.first.x > centerOfBack.x - side_dimension / 2 &&
            Back.first.z < centerOfBack.z + side_dimension / 2 && Back.first.z > centerOfBack.z - side_dimension / 2) {
            intersections[numIntersections] = Back;
            normals[numIntersections++] = normalOfBack;
        }


        // find the closest intersection to the camera (smallest t)
        // sort the intersections
        for (int i = 0; i < numIntersections; i++) {
            for (int j = i + 1; j < numIntersections; j++) {
                if (intersections[i].second > intersections[j].second) {
                    vec3 temp2 = normals[i];
                    normals[i] = normals[j];
                    normals[j] = temp2;
                    std::pair<vec3, float> temp = intersections[i];
                    intersections[i] = intersections[j];
                    intersections[j] = temp;
                }
            }
        }

        hit.t = intersections[1].second;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = normals[1] ;
        //todo normal vectors are shit here i guess
        return hit;
    }

    static std::pair<vec3, float> intersectionOfStraightLineAndPlane(vec3 start, vec3 dir, vec3 center, vec3 normal) {
        float f = dot(center - start, normal) / dot(dir, normal);
        vec3 intersectionPoint = start + dir * f;
        return std::make_pair(intersectionPoint, f);
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
    std::vector<Light *> lights;
    Camera camera;
    vec3 La;
public:
    void build() {
        vec3 eye = vec3(-0.7, 0.8, 0), vup = vec3(0, 0, 1), lookat = vec3(0, 0, 0);
        float fov = 45 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        La = vec3(0.0f, 0.0f, 0.0f);
        /*vec3 lightDirection(1, 1, 1), Le(1.5, 1.5, 1.5);
        lights.push_back(new Light(lightDirection, Le));*/

        vec3 kd(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
        for (int i = 0; i < 1; i++) {
            objects.push_back(new Sphere(vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f), rnd() * 0.1f));
            objects.push_back(new Cube(vec3(0.0f, 0.0f, 0.0f), rnd() * 0.5f));
        }
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
        vec3 outRadiance = vec3(0.2f,0.2f,0.2f);
        float specularAmbient = (0.2f * (1.0f + dot(hit.normal + vec3(0,0,0), ray.dir)));
        vec3 temp = vec3(outRadiance.x + specularAmbient, outRadiance.y + specularAmbient, outRadiance.z + specularAmbient);
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
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}	