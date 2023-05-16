//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Brosig Marton Janos
// Neptun : A0897X
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"
// A program elkeszitesehez a Minimal sugarkoveto CPU-n program forraskodjat hasznaltam fel.

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

    // A haromszog inteszekciojat a ray-tracing pdf 12. oldalan talalhato keplet alapjan szamoljuk.
    Hit intersect(const Ray &ray) override {
        Hit hit;
        float t = dot(r1 - ray.start, n) / dot(ray.dir, n);
        if (t <= 0) return hit;
        vec3 p = ray.start + ray.dir * t;
        // Megnezzuk hogy a pont a haromszogon belul van-e.
        if (dot(cross(r2 - r1, p - r1), n) > 0 &&
            dot(cross(r3 - r2, p - r2), n) > 0 &&
            dot(cross(r1 - r3, p - r3), n) > 0) {
            hit.t = t;
            hit.position = p;
            // Normal vektor iranyaval nem kell foglalkoznunk (elofordulhat hogy "rossz"
            // iranyba mutat, ekkor a firstIntersect fugveny majd megforditja.
            hit.normal = n;
        }
        return hit;
    }
};

struct Cube : public Intersectable {
    // Haromszogekbol epitjuk fel a kockat a obj definicio alapjan. Rendelkezesunkre bocsatjak a normal
    // vektorokat is, ezeket is felhasznaljuk.
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
                // Az elso 3 szam megmondja, hogy melyik pontok alkotnak egy lapot.
                // Az utolso parameter azt mondja meg, hogy milyen indexu normalvektor tartozik hozza. (index + 1)
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

        for (auto &face: faces) {
            // Az obj file segitsegevel egyesevel haromszogenkent hozzuk letre a kockat.
            // A side_length valtozo segitsegevel meretezzuk a kockat.
            // A center vallttozo segitsegevel eltolhatjuk a kockat.
            triangles.emplace_back(vertices[face[0] - 1] * side_length + center,
                                   vertices[face[1] - 1] * side_length + center,
                                   vertices[face[2] - 1] * side_length + center,
                                   normals[face[3] - 1]);
        }
    }

    Hit intersect(const Ray &ray) override {
        std::vector<Hit> hits;
        // Vegignezzuk a test osszes haromszoget, amelyeket metszunk, azokat eltaroljuk.
        for (auto &triangle: triangles) {
            Hit hit = triangle.intersect(ray);
            // Ervenytelen metszeseket nem tarolunk.
            if (hit.t > 0) {
                hits.push_back(hit);
            }
        }
        // Amennyiben nem talaltunk metszest, ures Hit-et adunk vissza, egyebkent megkeressuk
        // a tavolabbi metszespontot, ezzel elerjuk hogy mindig a "hatso" falakat lassuk,
        // ezzel atlatszo lesz a kocka hozzank kozelebb allo fala.
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
        // Az obj file-ban talalhato ertekeket hard-code-olva eltaroljuk.
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

        for (auto &face: faces) {
            // Letrehozunk egy haromszoget minden eltarolt lapra, a megadott meretezes es eltolas
            // segitsegevel.
            triangles.emplace_back(verticles[face[0] - 1] * scaling + center,
                                   verticles[face[1] - 1] * scaling + center,
                                   verticles[face[2] - 1] * scaling + center);
        }
    }

    Hit intersect(const Ray &ray) override {
        // Vegigmegyunk a haromszogeken, es megkeressuk a metszespontokat.
        std::vector<Hit> hits;
        for (auto &triangle: triangles) {
            Hit hit = triangle.intersect(ray);
            if (hit.t > 0) {
                hits.push_back(hit);
            }
        }
        // Amennyiben nem talaltunk metszest, ures Hit-et adunk vissza, egyebkent a legkozelebbit adjuk vissza.
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

    // Az obj file-ban talalhato ertekeket hard-code-olva eltaroljuk.
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

        for (auto &face: faces) {
            // Letrehozunk egy haromszoget minden eltarolt lapra, a megadott meretezes es eltolas
            // segitsegevel.
            triangles.emplace_back(verticles[face[0] - 1] * scaling + center,
                                   verticles[face[1] - 1] * scaling + center,
                                   verticles[face[2] - 1] * scaling + center);
        }
    }

    Hit intersect(const Ray &ray) override {
        // Vegigmegyunk a haromszogeken, es megkeressuk a metszespontokat.
        std::vector<Hit> hits;
        for (auto &triangle: triangles) {
            Hit hit = triangle.intersect(ray);
            if (hit.t > 0) {
                hits.push_back(hit);
            }
        }
        // Amennyiben nem talaltunk metszest, ures Hit-et adunk vissza, egyebkent a legkozelebbit adjuk vissza.
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

constexpr const float epsilon = 0.0001f;

struct Cone : Intersectable {
    // Az elnevezesek a ray-tracing pdf 11. diajan talalhato jeloleseket kovetik.
    Light *light;
    vec3 p; // a kup csucsa
    vec3 n; // egyseg vektor, mely a tengely iranyat mutatja
    float alfa; // bezart szog
    float h; // magassag

    Cone(vec3 p, vec3 n, float alfa, float h, Light *light) : light(light), p(p), n(n), alfa(alfa), h(h) {}

    Hit intersect(const Ray &ray) {
        // Kulon valtozoval taroljuk a sugar kezdo es vegpontjat, hogy konnyebben olvashato legyen a kod.
        vec3 s = ray.start;
        vec3 d = ray.dir;

        // Ezeket a kepleteket a ray-tracing pdf 11. diajan talalhato kepletek alapjan implementaltam.
        vec3 H = s - p;
        float a = pow(dot(d, n), 2.0f) - dot(d, d) * pow(cosf(alfa), 2.0f);
        float b = 2.0f * (dot(d, n) * dot(H, n) - dot(d, H) * pow(cosf(alfa), 2.0f));
        float c = pow(dot(H, n), 2.0f) - dot(H, H) * pow(cosf(alfa), 2.0f);

        // A diszkriminans es a ket lehetseges t ertek kiszamitasa.
        float D = b * b - 4.0f * a * c;
        if (D < 0.0f) return {};
        float sqrt_discr = sqrtf(D);
        // Eltaroljuk egy vectorban a ket lehetseges t erteket, es hogy melyik valid. (valid jelentese hogy a palast h alatti reszen van)
        std::vector<std::pair<float, bool>> ts = {
                {(-b + sqrt_discr) / (2.0f * a), false},
                {(-b - sqrt_discr) / (2.0f * a), false}
        };
        // Beallitjuk a valid valtozokat.
        for (auto &t: ts) {
            if (t.first > 0.0f) {
                Hit hit;
                hit.t = t.first;
                hit.position = s + hit.t * d;
                hit.normal = getNormal(hit);
                if (isValidHit(hit)) t.second = true;
            }
        }
        // A legkozelebbire van szuksegunk ami valid, ezert rendezzuk a vectorunket.
        if (ts[0].first > ts[1].first) {
            std::swap(ts[0], ts[1]);
        }
        Hit hit;
        // Ha az elso valid akkor azt valasztjuk, ha nem akkor megnezzuk, hogy a masodik valid-e
        // Ha az sem valid, akkor nincs metszespont, ures hitet adunk vissza.
        if (ts[0].second) {
            hit.t = ts[0].first;
            hit.position = s + hit.t * d;
            hit.normal = getNormal(hit);
            return hit;
        } else if (ts[1].second) {
            hit.t = ts[1].first;
            hit.position = s + hit.t * d;
            hit.normal = getNormal(hit);
            return hit;
        } else {
            return hit;
        }
    }

    // Igazat ad visza, ha a hit pozicioja a kup h magassaga alatt van.
    // a sugarkovetes pdf 11. diajan talalhato keplet alapjan implementaltam. (bekeretezett keplet)
    inline bool isValidHit(const Hit &hit) const {
        float dotProduct = dot(hit.position - p, n);
        return (0.0f <= dotProduct && dotProduct <= h);
    }

    inline vec3 getNormal(const Hit &hit) const {
        // b / cos(alfa/2) -> a befogobol az atfogo hossza (b a befogo)
        // length(hit.postion-p) -> hasonlo haromszogek, meghatarozzuk b hosszat, ezzel megkapjuk a
        // tengely iranyu vektort, amivel mar csak el kell tolnunk a hit.position - p-t, hogy egysegvektor legyen.
        return normalize(hit.position - p - (length(hit.position - p) / cosf(alfa)) * n);
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

class Scene {
    std::vector<Intersectable *> objects;
    std::vector<Cone *> cones;
    Camera camera;
    vec3 La;
public:
    // Milyen tavolsagra van a "megfigyelo" keszulek a kupok csucsatol (ertelemszeruen az iranyvektor iranyaba)
    constexpr const static float LIGHT_OFFSET_EPSILON = epsilon * 40;
    // A kupokat egy kicsit belesulyesztettem az eppen alatta levo objektumba,
    // igy termeszetesebben nez ki, es elkerulhetem a fenyeles miatti csillogo reszet a kup aljanak.
    constexpr const static float CONE_OFFSET_FROM_HIT_EPSILON = epsilon * 30;

    void build() {
        vec3 eye = vec3(1.0, 1.5f, -0.1f), vup = vec3(0, 0, 1), lookat = vec3(0, 0, 0);
        float fov = 45 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        // Hatterszinuk a feladatleiras szerint fekete.
        La = vec3(0.0f, 0.0f, 0.0f);

        // A kocka
        objects.push_back(new Cube(vec3(0.0f, 0.0f, 0.0f), 1));

        // A bal oldalon talahato ikozahedron.
        objects.push_back(new IcosaHedron(vec3(0.3f, 0.0f, -0.2f), 0.25f));

        // A jobb oldalon talalhato dodekahedron.
        objects.push_back(new DodecaHedron(vec3(-0.2f, 0.10f, -0.25f), 0.25f));

        // Nehany random koordinata a kupoknak.
        Hit hRedCone = firstIntersect(camera.getRay(230, 200));
        Hit hGreenCone = firstIntersect(camera.getRay(450, 500));
        Hit hBlueCone = firstIntersect(camera.getRay(324, 393));

        // Lehallgatokeszulekek letrehozasa.
        auto *redLight = new Light(hRedCone.position + hRedCone.normal * LIGHT_OFFSET_EPSILON, vec3(0.2f, 0.0f, 0.0f));
        auto *greenLight = new Light(hGreenCone.position + hGreenCone.normal * LIGHT_OFFSET_EPSILON,
                                     vec3(0.0f, 0.2f, 0.0f));
        auto *blueLight = new Light(hBlueCone.position + hBlueCone.normal * LIGHT_OFFSET_EPSILON,
                                    vec3(0.0f, 0.0f, 0.2f));

        // Mivel a kupokhoz egyertelmuen kapcsolodik egy lehallgato keszulek, ezert a kupok taroljak a lehallgato keszulekeiket.
        Cone *redCone = new Cone(hRedCone.position - hRedCone.normal * CONE_OFFSET_FROM_HIT_EPSILON,
                                 hRedCone.normal, 0.4, 0.1f, redLight);
        Cone *greenCone = new Cone(hGreenCone.position - hGreenCone.normal * CONE_OFFSET_FROM_HIT_EPSILON,
                                   hGreenCone.normal, 0.4f, 0.1f, greenLight);
        Cone *blueCone = new Cone(hBlueCone.position - hBlueCone.normal * CONE_OFFSET_FROM_HIT_EPSILON,
                                  hBlueCone.normal, 0.4f, 0.1f, blueLight);

        objects.push_back(redCone);
        objects.push_back(greenCone);
        objects.push_back(blueCone);

        cones.push_back(redCone);
        cones.push_back(greenCone);
        cones.push_back(blueCone);
    }

    void refresh(int pX, int pY) {
        // Kilovunk egy sugarat a felhasznalo altal kattintott pontba.
        Hit hit = firstIntersect(camera.getRay(pX, windowWidth - pY));
        float shortestDistance = INFINITY;
        Cone *closestCone = nullptr;

        // Megkeressuk a legkozelebbi kupot.
        for (auto cone: cones) {
            float currentLength = abs(length(hit.position - cone->p));
            if (currentLength < shortestDistance) {
                shortestDistance = currentLength;
                closestCone = cone;
            }
        }
        // Ha nincs ilyen kup, akkor visszaterunk.
        if (closestCone == nullptr) return;

        // Ha van ilyen kup, akkor athelyezzuk a benne levo lampat, es a kupot az uj helyere.
        closestCone->p = hit.position - hit.normal * CONE_OFFSET_FROM_HIT_EPSILON;
        closestCone->n = hit.normal;
        closestCone->light->position = hit.position + closestCone->n * LIGHT_OFFSET_EPSILON;
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
            if (hit.t > 0.0f && (bestHit.t < 0.0f || hit.t < bestHit.t)) bestHit = hit;
        }
        if (dot(ray.dir, bestHit.normal) > 0.0f) bestHit.normal = bestHit.normal * (-1.0f);
        return bestHit;
    }

    vec3 trace(Ray ray, int depth = 0) {
        Hit hit = firstIntersect(ray);
        // Ha nincsen talalat, akkor visszaterunk a hatter szinnel.
        if (hit.t < 0) return La;
        // A spekularis ambiens szinkomponens szamitasa.
        float ambientFactor = (0.2f * (1.0f + dot(hit.normal, ray.dir * (-1))));
        vec3 specularAmbient = vec3(ambientFactor, ambientFactor, ambientFactor);

        // Vegigmegyunk a lehallgatokeszulekek listajan, es megnezzuk, hogy a sugarunk eleri-e oket.
        // Ha egy lehallgatokeszulek elerheto a pontunkbol az ambiens fenyekhez hozzakeverjuk a lampa szineit.
        for (Cone *cone: cones) {
            // Kilovunk egy sugarat a pontunkbol a lehallgatokeszulek fele. (epsilon: hogy nehogy onmagat talalja el)
            Ray rayToLight = Ray(hit.position + hit.normal * epsilon,
                                 normalize(cone->light->position - hit.position));
            Hit hitToLight = firstIntersect(rayToLight);
            if (hitToLight.t < 0.0f) continue;
            // If the distance to the light is shorter than the distance to the hit, the light is visible.
            // Ha a kapott hit tavolsaga nagyobb, mint a feny tavolsaga, akkor a feny elerheto. (ezert emeltuk ki a kupbol egy kicsit a fenyt)
            if (length(hitToLight.position - hit.position) > length(cone->light->position - hit.position)) {
                // Hozzaadjuk a lehallgatokeszulek szinet az ambiens szinkomponenshez.
                // Figyelunk ra hogy nehogy 1 feletti erteket kapjunk.
                float multiplier = 1.0f / hitToLight.t;
                multiplier = multiplier > 3.0f ? 3.0f : multiplier;
                specularAmbient = specularAmbient + cone->light->Le * multiplier;
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
std::vector<vec4> image(windowWidth * windowHeight);

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();

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
            // Bal klikk eseten ujrarajzoljuk a kepet a valltoztatasok utan.
            scene.refresh(pX, pY);
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