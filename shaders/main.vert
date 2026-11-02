#version 450
#extension GL_EXT_scalar_block_layout : require

struct PointLight {
    vec4 position;
    vec4 color;
};

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 color;
    float inner_angle;
    float outer_angle;
};

layout(std430, set = 0, binding = 1) buffer PointLights {
    PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) buffer SpotLights {
    SpotLight spot_lights[];
};

layout(set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec4 camera_position;
    vec4 dir_light_direction;
    vec4 dir_light_color;
    vec4 ambient_light;
    int point_light_count;
    int spot_light_count;
} scene;

layout(set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec4 albedo;
    vec4 specular;
    float shininess;
} model;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragUV;

void main() {
    vec4 worldPos = model.model * vec4(inPosition, 1.0);
    fragWorldPos = worldPos.xyz;
    
    mat3 normalMatrix = transpose(inverse(mat3(model.model)));
    fragNormal = normalize(normalMatrix * inNormal);
    fragUV = inUV;
    
    gl_Position = scene.view_projection * worldPos;
}