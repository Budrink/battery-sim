#include "battery_native/renderer.hpp"
#include "battery_native/cuda_gl_field.hpp"
#include "battery_native/volume_shader.hpp"

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <vector>

namespace battery {
namespace {

constexpr float pi = 3.14159265358979323846f;

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
};

struct Mesh {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLsizei index_count = 0;
    GLenum mode = GL_TRIANGLES;

    void upload(const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices,
                GLenum draw_mode = GL_TRIANGLES) {
        mode = draw_mode;
        index_count = static_cast<GLsizei>(indices.size());
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ebo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex)),
                     vertices.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(indices.size() * sizeof(unsigned int)),
                     indices.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                              reinterpret_cast<void*>(offsetof(Vertex, position)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                              reinterpret_cast<void*>(offsetof(Vertex, normal)));
        glBindVertexArray(0);
    }

    void destroy() {
        if (ebo) glDeleteBuffers(1, &ebo);
        if (vbo) glDeleteBuffers(1, &vbo);
        if (vao) glDeleteVertexArrays(1, &vao);
        vao = vbo = ebo = 0;
        index_count = 0;
    }
};

GLuint compile_shader(GLenum type, const char* source) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048]{};
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::fprintf(stderr, "battery renderer shader error:\n%s\n", log);
    }
    return shader;
}

GLuint create_program() {
    static constexpr const char* vertex_source = R"glsl(
#version 330 core
layout(location=0) in vec3 aPosition;
layout(location=1) in vec3 aNormal;
layout(location=2) in mat4 aInstanceModel;
layout(location=6) in vec4 aInstanceColor;
uniform bool uInstanced;
uniform mat4 uViewProjection;
uniform mat4 uMVP;
uniform mat4 uModel;
out vec3 vNormal;
out vec3 vWorld;
out vec4 vInstanceColor;
void main() {
    mat4 model = uInstanced ? aInstanceModel : uModel;
    vec4 world = model * vec4(aPosition, 1.0);
    vWorld = world.xyz;
    vNormal = normalize(mat3(transpose(inverse(model))) * aNormal);
    vInstanceColor = aInstanceColor;
    gl_Position = uInstanced ? uViewProjection * world : uMVP * vec4(aPosition, 1.0);
}
)glsl";
    static constexpr const char* fragment_source = R"glsl(
#version 330 core
in vec3 vNormal;
in vec3 vWorld;
in vec4 vInstanceColor;
uniform bool uInstanced;
out vec4 fragColor;
uniform vec4 uColor;
uniform vec3 uEye;
uniform vec3 uLight;
uniform float uEmissive;
uniform float uFresnel;
void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uEye - vWorld);
    float diffuse = max(dot(N, normalize(uLight)), 0.0);
    float fill = 0.30 + 0.70 * diffuse;
    float rim = pow(1.0 - abs(dot(N, V)), 2.2);
    // Lighting may dim the scalar color, but must not clip its channels and
    // turn orange/yellow into white before the top of the temperature scale.
    float lighting = mix(fill, 1.0, clamp(uEmissive, 0.0, 1.0));
    vec4 color = uInstanced ? vInstanceColor : uColor;
    vec3 rgb = color.rgb * min(lighting + rim * uFresnel, 1.0);
    float alpha = color.a * (1.0 + rim * uFresnel * 0.55);
    fragColor = vec4(rgb, clamp(alpha, 0.0, 1.0));
}
)glsl";
    const GLuint vertex = compile_shader(GL_VERTEX_SHADER, vertex_source);
    const GLuint fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048]{};
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        std::fprintf(stderr, "battery renderer program error:\n%s\n", log);
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

Mesh make_cube() {
    const std::array<glm::vec3, 8> corners = {
        glm::vec3{-0.5f,-0.5f,-0.5f}, glm::vec3{ 0.5f,-0.5f,-0.5f},
        glm::vec3{ 0.5f, 0.5f,-0.5f}, glm::vec3{-0.5f, 0.5f,-0.5f},
        glm::vec3{-0.5f,-0.5f, 0.5f}, glm::vec3{ 0.5f,-0.5f, 0.5f},
        glm::vec3{ 0.5f, 0.5f, 0.5f}, glm::vec3{-0.5f, 0.5f, 0.5f},
    };
    struct Face { int a, b, c, d; glm::vec3 normal; };
    const std::array<Face, 6> faces = {{
        {0,3,2,1,{0,0,-1}}, {4,5,6,7,{0,0,1}},
        {0,1,5,4,{0,-1,0}}, {3,7,6,2,{0,1,0}},
        {0,4,7,3,{-1,0,0}}, {1,2,6,5,{1,0,0}},
    }};
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    for (const auto& face : faces) {
        const unsigned int base = static_cast<unsigned int>(vertices.size());
        vertices.push_back({corners[face.a], face.normal});
        vertices.push_back({corners[face.b], face.normal});
        vertices.push_back({corners[face.c], face.normal});
        vertices.push_back({corners[face.d], face.normal});
        indices.insert(indices.end(), {base, base+1, base+2, base, base+2, base+3});
    }
    Mesh mesh;
    mesh.upload(vertices, indices);
    return mesh;
}

Mesh make_box_outline() {
    std::vector<Vertex> vertices;
    for(int z=0;z<2;++z)for(int y=0;y<2;++y)for(int x=0;x<2;++x)
        vertices.push_back({{x-.5f,y-.5f,z-.5f},{0,0,1}});
    std::vector<unsigned int> indices;
    for(unsigned i=0;i<8;++i)for(unsigned bit:{1u,2u,4u})if(!(i&bit)) {
        indices.push_back(i);indices.push_back(i|bit);
    }
    Mesh mesh;mesh.upload(vertices,indices,GL_LINES);return mesh;
}

Mesh make_cylinder(int segments = 48) {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    for (int index = 0; index <= segments; ++index) {
        const float angle = 2.0f * pi * static_cast<float>(index) / static_cast<float>(segments);
        const float x = 0.5f * std::cos(angle);
        const float y = 0.5f * std::sin(angle);
        const glm::vec3 normal = glm::normalize(glm::vec3{x, y, 0.0f});
        vertices.push_back({{x, y, -0.5f}, normal});
        vertices.push_back({{x, y,  0.5f}, normal});
    }
    for (int index = 0; index < segments; ++index) {
        const unsigned int base = static_cast<unsigned int>(index * 2);
        indices.insert(indices.end(), {base, base+1, base+3, base, base+3, base+2});
    }
    for (int cap = 0; cap < 2; ++cap) {
        const float z = cap == 0 ? -0.5f : 0.5f;
        const glm::vec3 normal{0.0f, 0.0f, cap == 0 ? -1.0f : 1.0f};
        const unsigned int center = static_cast<unsigned int>(vertices.size());
        vertices.push_back({{0,0,z}, normal});
        for (int index = 0; index <= segments; ++index) {
            const float angle = 2.0f * pi * static_cast<float>(index) / static_cast<float>(segments);
            vertices.push_back({{0.5f*std::cos(angle), 0.5f*std::sin(angle), z}, normal});
        }
        for (int index = 0; index < segments; ++index) {
            const unsigned int a = center + 1 + static_cast<unsigned int>(index);
            const unsigned int b = a + 1;
            if (cap == 0) indices.insert(indices.end(), {center, b, a});
            else indices.insert(indices.end(), {center, a, b});
        }
    }
    Mesh mesh;
    mesh.upload(vertices, indices);
    return mesh;
}

Mesh make_cone(int segments = 36) {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    for (int index = 0; index <= segments; ++index) {
        const float angle = 2.0f * pi * static_cast<float>(index) / static_cast<float>(segments);
        const float x = 0.5f * std::cos(angle);
        const float y = 0.5f * std::sin(angle);
        const glm::vec3 side_normal = glm::normalize(glm::vec3{x, y, 0.25f});
        vertices.push_back({{x, y, -0.5f}, side_normal});
        vertices.push_back({{0.0f, 0.0f, 0.5f}, side_normal});
    }
    for (int index = 0; index < segments; ++index) {
        const unsigned int base = static_cast<unsigned int>(index * 2);
        indices.insert(indices.end(), {base, base + 2, base + 1});
    }
    const unsigned int center = static_cast<unsigned int>(vertices.size());
    vertices.push_back({{0,0,-0.5f}, {0,0,-1}});
    for (int index = 0; index <= segments; ++index) {
        const float angle = 2.0f * pi * static_cast<float>(index) / static_cast<float>(segments);
        vertices.push_back({{0.5f*std::cos(angle), 0.5f*std::sin(angle), -0.5f}, {0,0,-1}});
    }
    for (int index = 0; index < segments; ++index) {
        const unsigned int a = center + 1 + static_cast<unsigned int>(index);
        indices.insert(indices.end(), {center, a + 1, a});
    }
    Mesh mesh;
    mesh.upload(vertices, indices);
    return mesh;
}

Mesh make_sphere(int slices = 32, int stacks = 20) {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    for (int stack = 0; stack <= stacks; ++stack) {
        const float v = static_cast<float>(stack) / static_cast<float>(stacks);
        const float phi = pi * (v - 0.5f);
        const float z = 0.5f * std::sin(phi);
        const float radius = 0.5f * std::cos(phi);
        for (int slice = 0; slice <= slices; ++slice) {
            const float theta = 2.0f * pi * static_cast<float>(slice) / static_cast<float>(slices);
            const glm::vec3 position{radius*std::cos(theta), radius*std::sin(theta), z};
            vertices.push_back({position, glm::normalize(position)});
        }
    }
    for (int stack = 0; stack < stacks; ++stack) {
        for (int slice = 0; slice < slices; ++slice) {
            const unsigned int a = static_cast<unsigned int>(stack * (slices+1) + slice);
            const unsigned int b = a + static_cast<unsigned int>(slices+1);
            indices.insert(indices.end(), {a,b,a+1,a+1,b,b+1});
        }
    }
    Mesh mesh;
    mesh.upload(vertices, indices);
    return mesh;
}

glm::vec3 to_glm(const Vec3& value) {
    return {static_cast<float>(value.x), static_cast<float>(value.y),
            static_cast<float>(value.z)};
}

glm::vec3 to_glm_rgb(const Rgb& color) {
    return {color[0], color[1], color[2]};
}

glm::vec3 degradation_color(float normalized) {
    return to_glm_rgb(degradation_rgb(normalized));
}

// Deliberately outside both scalar palettes: unmistakable burnt/destroyed state.
constexpr glm::vec3 destroyed_color{destroyed_rgb[0], destroyed_rgb[1], destroyed_rgb[2]};

glm::mat4 cell_transform(const CellConfig& cell, float inset = 1.0f) {
    const auto& q = cell.orientation_xyzw;
    glm::mat4 model = glm::translate(glm::mat4(1.0f), to_glm(cell.position_m));
    model *= glm::toMat4(glm::normalize(glm::quat(
        static_cast<float>(q[3]), static_cast<float>(q[0]),
        static_cast<float>(q[1]), static_cast<float>(q[2]))));
    if (cell.shape == ShapeKind::Cylinder) {
        const float diameter = static_cast<float>(cell.dimensions_m[0] * 2.0) * inset;
        model = glm::scale(model, {diameter, diameter,
            static_cast<float>(cell.dimensions_m[1]) * inset});
    } else {
        model = glm::scale(model, {
            static_cast<float>(cell.dimensions_m[0]) * inset,
            static_cast<float>(cell.dimensions_m[1]) * inset,
            static_cast<float>(cell.dimensions_m[2]) * inset,
        });
    }
    return model;
}

glm::vec3 probe_color(std::size_t index) {
    static constexpr std::array<glm::vec3, 8> colors = {{
        {1.00f,0.38f,0.36f}, {0.38f,1.00f,0.48f},
        {0.30f,0.66f,1.00f}, {1.00f,0.86f,0.30f},
        {0.86f,0.38f,1.00f}, {0.30f,1.00f,0.94f},
        {1.00f,0.58f,0.18f}, {0.80f,0.80f,0.80f},
    }};
    return colors[index % colors.size()];
}

struct SceneFrame {
    glm::vec3 minimum{0};
    glm::vec3 maximum{0};
    glm::vec3 center{0};
    glm::vec3 eye{0};
    float extent = 1.0f;
    glm::mat4 view_projection{1.0f};
};

SceneFrame make_scene_frame(const Scenario& scenario, const OrbitCamera& camera,
                            int width, int height) {
    SceneFrame frame;
    frame.minimum = glm::vec3(std::numeric_limits<float>::max());
    frame.maximum = glm::vec3(-std::numeric_limits<float>::max());
    for (const auto& cell : scenario.cells) {
        const glm::vec3 position = to_glm(cell.position_m);
        glm::vec3 half;
        if (cell.shape == ShapeKind::Cylinder) {
            half = {static_cast<float>(cell.dimensions_m[0]),
                    static_cast<float>(cell.dimensions_m[0]),
                    static_cast<float>(cell.dimensions_m[1] * 0.5)};
        } else {
            half = {static_cast<float>(cell.dimensions_m[0] * 0.5),
                    static_cast<float>(cell.dimensions_m[1] * 0.5),
                    static_cast<float>(cell.dimensions_m[2] * 0.5)};
        }
        frame.minimum = glm::min(frame.minimum, position - half);
        frame.maximum = glm::max(frame.maximum, position + half);
    }
    if (scenario.cells.empty()) {
        frame.minimum = {-0.5f,-0.5f,-0.5f};
        frame.maximum = { 0.5f, 0.5f, 0.5f};
    }
    frame.center = 0.5f * (frame.minimum + frame.maximum);
    frame.extent = std::max(glm::length(frame.maximum - frame.minimum), 0.05f);
    const float cp = std::cos(camera.pitch);
    const glm::vec3 direction{
        cp * std::cos(camera.yaw), cp * std::sin(camera.yaw), std::sin(camera.pitch)};
    frame.eye = frame.center + direction * frame.extent * camera.distance;
    const glm::mat4 view = glm::lookAt(frame.eye, frame.center, glm::vec3{0,0,1});
    const glm::mat4 projection = glm::perspective(glm::radians(42.0f),
        static_cast<float>(std::max(width, 1)) / static_cast<float>(std::max(height, 1)),
        frame.extent * 0.002f, frame.extent * 30.0f);
    frame.view_projection = projection * view;
    return frame;
}

bool ray_unit_box(const glm::vec3& origin, const glm::vec3& direction, float& hit_t) {
    float near_t = 0.0f;
    float far_t = std::numeric_limits<float>::max();
    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(direction[axis]) < 1.0e-8f) {
            if (origin[axis] < -0.5f || origin[axis] > 0.5f) return false;
            continue;
        }
        float a = (-0.5f - origin[axis]) / direction[axis];
        float b = ( 0.5f - origin[axis]) / direction[axis];
        if (a > b) std::swap(a, b);
        near_t = std::max(near_t, a);
        far_t = std::min(far_t, b);
        if (near_t > far_t) return false;
    }
    hit_t = near_t;
    return far_t >= 0.0f;
}

}  // namespace

struct SceneRenderer::Impl {
    std::unique_ptr<CudaGlField> gpu_colors;
    const ThermalMesh* cached_mesh = nullptr;
    std::size_t cached_volume_count = 0;
    float cached_cutaway = -1;
    GLsizei instance_count = 0;
    GLuint instance_buffer = 0;
    GLuint color_buffer = 0;
    GLuint volume_program = 0, sample_texture = 0;
    GLuint palettes[2]{};
    GLint u_instanced = -1;
    GLint u_view_projection = -1;
    GLuint program = 0;
    GLuint fbo = 0;
    GLuint color_texture = 0;
    GLuint depth_buffer = 0;
    int fbo_width = 0;
    int fbo_height = 0;
    Mesh cube;
    Mesh box_outline;
    Mesh cylinder;
    Mesh cone;
    Mesh sphere;
    GLint u_mvp = -1;
    GLint u_model = -1;
    GLint u_color = -1;
    GLint u_eye = -1;
    GLint u_light = -1;
    GLint u_emissive = -1;
    GLint u_fresnel = -1;

    void ensure_fbo(int width, int height) {
        if (fbo && width == fbo_width && height == fbo_height) return;
        if (fbo) {
            glDeleteFramebuffers(1, &fbo);
            glDeleteTextures(1, &color_texture);
            glDeleteRenderbuffers(1, &depth_buffer);
        }
        fbo_width = width;
        fbo_height = height;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glGenTextures(1, &color_texture);
        glBindTexture(GL_TEXTURE_2D, color_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, color_texture, 0);
        glGenRenderbuffers(1, &depth_buffer);
        glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                  GL_RENDERBUFFER, depth_buffer);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void draw(const Mesh& mesh, const glm::mat4& model, const glm::mat4& view_projection,
              const glm::vec4& color, const glm::vec3& eye,
              float emissive, float fresnel) const {
        glUniform1i(u_instanced, GL_FALSE);
        const glm::mat4 mvp = view_projection * model;
        glUniformMatrix4fv(u_mvp, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniformMatrix4fv(u_model, 1, GL_FALSE, glm::value_ptr(model));
        glUniform4fv(u_color, 1, glm::value_ptr(color));
        glUniform3fv(u_eye, 1, glm::value_ptr(eye));
        glUniform1f(u_emissive, emissive);
        glUniform1f(u_fresnel, fresnel);
        glBindVertexArray(mesh.vao);
        glDrawElements(mesh.mode, mesh.index_count, GL_UNSIGNED_INT, nullptr);
    }
};

SceneRenderer::SceneRenderer() : impl_(std::make_unique<Impl>()) {}
SceneRenderer::~SceneRenderer() { destroy(); }

bool SceneRenderer::init() {
    impl_->program = create_program();
    if (!impl_->program) return false;
    impl_->u_instanced = glGetUniformLocation(impl_->program, "uInstanced");
    impl_->u_view_projection = glGetUniformLocation(impl_->program, "uViewProjection");
    glGenBuffers(1, &impl_->instance_buffer);
    glGenBuffers(1, &impl_->color_buffer);
    impl_->gpu_colors = std::make_unique<CudaGlField>();
    const GLuint vs=compile_shader(GL_VERTEX_SHADER,volume_vertex_shader);
    const GLuint fs=compile_shader(GL_FRAGMENT_SHADER,volume_fragment_shader);
    impl_->volume_program=glCreateProgram();
    glAttachShader(impl_->volume_program,vs);glAttachShader(impl_->volume_program,fs);
    glLinkProgram(impl_->volume_program);glDeleteShader(vs);glDeleteShader(fs);
    GLint linked=0;glGetProgramiv(impl_->volume_program,GL_LINK_STATUS,&linked);
    if(!linked)return false;
    glGenTextures(1,&impl_->sample_texture);
    glGenTextures(2,impl_->palettes);
    for(int layer=0;layer<2;++layer) {
        std::vector<float> colors;
        for(int i=0;i<256;++i) {
            auto c=field_rgb(static_cast<FieldLayer>(layer),i/255.f,false);
            colors.insert(colors.end(),c.begin(),c.end());
        }
        glBindTexture(GL_TEXTURE_1D,impl_->palettes[layer]);
        glTexImage1D(GL_TEXTURE_1D,0,GL_RGB32F,256,0,GL_RGB,GL_FLOAT,colors.data());
        glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    }
    impl_->u_mvp = glGetUniformLocation(impl_->program, "uMVP");
    impl_->u_model = glGetUniformLocation(impl_->program, "uModel");
    impl_->u_color = glGetUniformLocation(impl_->program, "uColor");
    impl_->u_eye = glGetUniformLocation(impl_->program, "uEye");
    impl_->u_light = glGetUniformLocation(impl_->program, "uLight");
    impl_->u_emissive = glGetUniformLocation(impl_->program, "uEmissive");
    impl_->u_fresnel = glGetUniformLocation(impl_->program, "uFresnel");
    impl_->cube = make_cube();
    impl_->box_outline = make_box_outline();
    impl_->cylinder = make_cylinder();
    impl_->cone = make_cone();
    impl_->sphere = make_sphere();
    return true;
}

unsigned int SceneRenderer::render(int width, int height, const Scenario& scenario,
                                   const Simulation& simulation, const OrbitCamera& camera,
                                   const VisualSettings& settings, int selected_cell,
                                   const std::vector<int>& probed_cells) {
    if (!impl_->program || width <= 0 || height <= 0) return 0;
    impl_->ensure_fbo(width, height);

    const SceneFrame frame = make_scene_frame(scenario, camera, width, height);
    const glm::vec3& eye = frame.eye;
    const float extent = frame.extent;
    const glm::mat4& view_projection = frame.view_projection;

    glBindFramebuffer(GL_FRAMEBUFFER, impl_->fbo);
    glViewport(0, 0, width, height);
    glClearColor(0.012f, 0.012f, 0.016f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(impl_->program);
    const glm::vec3 light = glm::normalize(glm::vec3{-0.35f, -0.55f, 0.78f});
    glUniform3fv(impl_->u_light, 1, glm::value_ptr(light));

    if (settings.show_interfaces) {
        for (std::size_t index = 0; index < scenario.interfaces.size(); ++index) {
            const auto& edge = scenario.interfaces[index];
            const auto& state = simulation.interfaces()[index];
            const glm::vec3 a = to_glm(edge.point_a_m);
            const glm::vec3 b = to_glm(edge.point_b_m);
            const glm::vec3 delta = b - a;
            const float length = glm::length(delta);
            if (length <= 1.0e-8f) continue;
            const float ratio = static_cast<float>(state.l_m / std::max(state.l0_m, 1.0e-12));
            const float activation = 1.0f / (1.0f + std::exp(-7.0f * (ratio - 1.0f)));
            const glm::quat rotation = glm::rotation(glm::vec3{0,0,1}, glm::normalize(delta));
            float patch_side = extent * 0.10f;
            for (const auto& port : scenario.ports) {
                if (port.id == edge.port_a || port.id == edge.port_b) {
                    patch_side = std::max(static_cast<float>(std::sqrt(port.area_m2)),
                                          extent * 0.025f);
                    break;
                }
            }

            // The interface is real material, so it participates in the same
            // temperature field. Slices interpolate only inside that material;
            // no glyph is emitted into empty space.
            glDepthMask(GL_FALSE);
            constexpr int slices = 14;
            const float ta = static_cast<float>(
                simulation.cells()[edge.cell_a].temperature_k);
            const float tb = static_cast<float>(
                simulation.cells()[edge.cell_b].temperature_k);
            for (int slice = 0; slice < slices; ++slice) {
                const float u0 = static_cast<float>(slice) / static_cast<float>(slices);
                const float u1 = static_cast<float>(slice + 1) / static_cast<float>(slices);
                const float um = 0.5f * (u0 + u1);
                const float temperature = ta + (tb - ta) * um;
                const float normalized = (temperature - settings.temperature_min_k) /
                    std::max(settings.temperature_max_k - settings.temperature_min_k, 1.0f);
                const glm::vec3 field_color = to_glm_rgb(field_rgb(settings.field_layer,
                    settings.field_layer == FieldLayer::Temperature
                        ? normalized : static_cast<float>(state.damage),
                    state.phase == EdgePhase::Destroyed));
                glm::mat4 model = glm::translate(glm::mat4(1), a + delta * um);
                model *= glm::toMat4(rotation);
                model = glm::scale(model, {patch_side, patch_side,
                    length / static_cast<float>(slices)});
                impl_->draw(impl_->cube, model, view_projection,
                            {field_color, settings.field_opacity * 0.90f},
                            eye, 0.85f, 0.06f);
            }

            // l/l0 remains a property of the element, shown on its boundary,
            // not as a fictitious object floating beside the contour.
            glm::vec3 edge_color = settings.field_layer == FieldLayer::Temperature
                ? glm::mix(glm::vec3{0.20f,0.21f,0.23f},
                           glm::vec3{0.55f,0.53f,0.49f}, activation)
                : degradation_color(static_cast<float>(state.damage));
            if (state.phase == EdgePhase::Destroyed) edge_color = destroyed_color;
            glm::mat4 outline_model = glm::translate(glm::mat4(1), 0.5f * (a+b));
            outline_model *= glm::toMat4(rotation);
            outline_model = glm::scale(outline_model, {patch_side, patch_side,
                length});
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glLineWidth(state.phase == EdgePhase::Destroyed ? 2.6f : 1.4f);
            impl_->draw(impl_->box_outline, outline_model, view_projection,
                        {edge_color, state.phase == EdgePhase::Destroyed ? 0.96f : 0.50f},
                        eye, 0.80f, 0.12f);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glDepthMask(GL_TRUE);
        }
    }

    // One continuous optical volume per physical object. The solver's zone
    // boundaries have no optical walls, gaps or per-zone opacity.
    const auto& mesh = simulation.thermal_mesh();
    if (impl_->cached_mesh != &mesh || impl_->cached_volume_count != mesh.volumes.size()) {
        impl_->gpu_colors->detach();
        std::vector<int> ids(mesh.volumes.size());
        std::iota(ids.begin(),ids.end(),0);
        std::vector<float> aging;
        for(const auto& v:mesh.volumes)
            aging.push_back(static_cast<float>(scenario.cells[v.object].aging_fraction));
        glBindBuffer(GL_ARRAY_BUFFER,impl_->color_buffer);
        glBufferData(GL_ARRAY_BUFFER,static_cast<GLsizeiptr>(ids.size()*sizeof(glm::vec4)),nullptr,GL_DYNAMIC_DRAW);
        impl_->gpu_colors->attach(impl_->color_buffer,ids,aging);
        glBindTexture(GL_TEXTURE_BUFFER,impl_->sample_texture);
        glTexBuffer(GL_TEXTURE_BUFFER,GL_RGBA32F,impl_->color_buffer);
        impl_->cached_mesh=&mesh;
        impl_->cached_volume_count=mesh.volumes.size();
    }
    impl_->gpu_colors->update_samples(simulation.cuda_field());
    const GLuint vp=impl_->volume_program;
    glUseProgram(vp);
    auto uniform=[&](const char* name){return glGetUniformLocation(vp,name);};
    glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_BUFFER,impl_->sample_texture);
    glUniform1i(uniform("uSamples"),0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_1D,impl_->palettes[settings.field_layer==FieldLayer::Temperature?0:1]);
    glUniform1i(uniform("uPalette"),1);
    glUniform1i(uniform("uLayer"),static_cast<int>(settings.field_layer));
    glUniform2f(uniform("uRange"),settings.temperature_min_k,settings.temperature_max_k);
    glUniform1f(uniform("uOpacity"),std::clamp(settings.field_opacity,0.f,1.f));
    glUniform1f(uniform("uCut"),std::clamp(settings.cutaway_fraction,0.001f,1.f));
    glUniform3fv(uniform("uDestroyed"),1,glm::value_ptr(destroyed_color));
    // Transparent, non-overlapping macro volumes: back-to-front compositing.
    std::vector<std::size_t> order(scenario.cells.size());std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](auto a,auto b) {
        return glm::length(to_glm(scenario.cells[a].position_m)-eye)>
               glm::length(to_glm(scenario.cells[b].position_m)-eye);
    });
    glDepthMask(GL_FALSE);glEnable(GL_CULL_FACE);glCullFace(GL_FRONT);
    glBindVertexArray(impl_->cube.vao);
    for(auto index:order) {
        const auto& cell=scenario.cells[index];
        const auto model=cell_transform(cell);
        const auto mvp=view_projection*model;
        const auto local_eye=glm::vec3(glm::inverse(model)*glm::vec4(eye,1));
        glUniformMatrix4fv(uniform("uMVP"),1,GL_FALSE,glm::value_ptr(mvp));
        glUniform3fv(uniform("uLocalEye"),1,glm::value_ptr(local_eye));
        glUniform3i(uniform("uGrid"),cell.thermal_divisions[0],cell.thermal_divisions[1],cell.thermal_divisions[2]);
        glUniform1i(uniform("uOffset"),mesh.object_offsets[index]);
        glDrawElements(GL_TRIANGLES,impl_->cube.index_count,GL_UNSIGNED_INT,nullptr);
    }
    glDisable(GL_CULL_FACE);glActiveTexture(GL_TEXTURE0);
    glUseProgram(impl_->program);

    glDepthMask(GL_FALSE);
    for (std::size_t index = 0; index < scenario.cells.size(); ++index) {
        const auto& cell = scenario.cells[index];
        const auto& state = simulation.cells()[index];
        const float combined_degradation = static_cast<float>(
            1.0 - (1.0 - std::clamp(cell.aging_fraction, 0.0, 1.0)) *
                  (1.0 - std::clamp(state.damage, 0.0, 1.0)));
        const Mesh& mesh = cell.shape == ShapeKind::Cylinder ? impl_->cylinder : impl_->cube;
        const float selected = static_cast<int>(index) == selected_cell ? 1.0f : 0.0f;
        auto probe_it = std::find(probed_cells.begin(), probed_cells.end(),
                                  static_cast<int>(index));
        const bool probed = probe_it != probed_cells.end();
        glm::vec3 outline = probed
            ? probe_color(static_cast<std::size_t>(std::distance(probed_cells.begin(), probe_it)))
            : glm::mix(glm::vec3{0.45f,0.47f,0.49f}, glm::vec3{0.92f}, selected);
        // Destruction stays visible even when a cell is selected/probed.
        if (state.mode == CellMode::Spent) outline = destroyed_color;
        else if (state.mode == CellMode::Runaway && !probed)
            outline = settings.field_layer == FieldLayer::Temperature
                ? glm::vec3{1.0f,0.86f,0.32f}
                : degradation_color(combined_degradation);
        impl_->draw(mesh, cell_transform(cell), view_projection,
                    {glm::vec3{0.72f,0.73f,0.74f},
                     settings.shell_opacity + selected*0.06f},
                    eye, 0.08f, 1.45f + selected);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glLineWidth(probed ? 2.8f : (state.mode != CellMode::Healthy ? 2.6f :
                    (selected > 0.0f ? 2.3f : 1.0f)));
        impl_->draw(cell.shape == ShapeKind::Cylinder ? mesh : impl_->box_outline,
                    cell_transform(cell, 1.006f), view_projection,
                    {outline, state.mode == CellMode::Spent || probed
                        ? 0.96f : 0.30f + selected*0.48f},
                    eye, state.mode == CellMode::Spent || probed ? 0.9f : 0.42f, 0.20f);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    }

    if (settings.show_ports) {
        const float diameter = extent * 0.025f;
        for (const auto& port : scenario.ports) {
            const glm::vec3 color = port.kind == "vent"
                ? glm::vec3{1.0f,0.24f,0.04f} : glm::vec3{0.1f,0.85f,0.95f};
            const glm::mat4 model = glm::translate(glm::mat4(1), to_glm(port.centroid_m)) *
                glm::scale(glm::mat4(1), glm::vec3(diameter));
            impl_->draw(impl_->sphere, model, view_projection,
                        {color,0.88f}, eye, 1.1f, 0.6f);
        }
    }
    glDepthMask(GL_TRUE);
    glLineWidth(1.0f);
    glBindVertexArray(0);
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return impl_->color_texture;
}

int SceneRenderer::pick_cell(float x_px, float y_px, int width, int height,
                             const Scenario& scenario, const OrbitCamera& camera) const {
    if (width <= 0 || height <= 0 || scenario.cells.empty()) return -1;
    const SceneFrame frame = make_scene_frame(scenario, camera, width, height);
    const float ndc_x = 2.0f * x_px / static_cast<float>(width) - 1.0f;
    const float ndc_y = 1.0f - 2.0f * y_px / static_cast<float>(height);
    const glm::mat4 inverse_vp = glm::inverse(frame.view_projection);
    glm::vec4 near_point = inverse_vp * glm::vec4(ndc_x, ndc_y, -1.0f, 1.0f);
    glm::vec4 far_point = inverse_vp * glm::vec4(ndc_x, ndc_y, 1.0f, 1.0f);
    near_point /= near_point.w;
    far_point /= far_point.w;
    const glm::vec3 ray_origin = glm::vec3(near_point);
    const glm::vec3 ray_direction = glm::normalize(glm::vec3(far_point - near_point));

    int best_cell = -1;
    float best_t = std::numeric_limits<float>::max();
    for (std::size_t index = 0; index < scenario.cells.size(); ++index) {
        const glm::mat4 inverse_model = glm::inverse(cell_transform(scenario.cells[index]));
        const glm::vec3 local_origin = glm::vec3(inverse_model * glm::vec4(ray_origin, 1.0f));
        const glm::vec3 local_direction = glm::vec3(
            inverse_model * glm::vec4(ray_direction, 0.0f));
        float hit_t = 0.0f;
        if (ray_unit_box(local_origin, local_direction, hit_t) &&
            hit_t >= 0.0f && hit_t < best_t) {
            best_t = hit_t;
            best_cell = static_cast<int>(index);
        }
    }
    return best_cell;
}

void SceneRenderer::destroy() {
    if (!impl_) return;
    impl_->gpu_colors.reset();
    impl_->cached_mesh = nullptr;
    impl_->cached_volume_count = 0;
    impl_->cached_cutaway = -1;
    impl_->cube.destroy();
    impl_->box_outline.destroy();
    impl_->cylinder.destroy();
    impl_->cone.destroy();
    impl_->sphere.destroy();
    if (impl_->instance_buffer) glDeleteBuffers(1, &impl_->instance_buffer);
    impl_->instance_buffer = 0;
    if (impl_->color_buffer) glDeleteBuffers(1, &impl_->color_buffer);
    impl_->color_buffer = 0;
    if (impl_->sample_texture) glDeleteTextures(1,&impl_->sample_texture);
    glDeleteTextures(2,impl_->palettes);
    if (impl_->volume_program) glDeleteProgram(impl_->volume_program);
    impl_->sample_texture=impl_->volume_program=0;
    impl_->palettes[0]=impl_->palettes[1]=0;
    if (impl_->program) glDeleteProgram(impl_->program);
    if (impl_->fbo) glDeleteFramebuffers(1, &impl_->fbo);
    if (impl_->color_texture) glDeleteTextures(1, &impl_->color_texture);
    if (impl_->depth_buffer) glDeleteRenderbuffers(1, &impl_->depth_buffer);
    impl_->program = impl_->fbo = impl_->color_texture = impl_->depth_buffer = 0;
}

}  // namespace battery
