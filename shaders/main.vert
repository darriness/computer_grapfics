#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

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

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;

void main() {
    vec4 worldPos = model.model * vec4(inPosition, 1.0);
    fragWorldPos = worldPos.xyz;
    
    // Правильное преобразование нормалей (матрица обратная транспонированная)
    mat3 normalMatrix = transpose(inverse(mat3(model.model)));
    fragNormal = normalize(normalMatrix * inNormal);
    
    gl_Position = scene.view_projection * worldPos;
}