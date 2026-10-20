#version 450
#extension GL_EXT_scalar_block_layout : require

struct PointLight {
    vec4 position; // xyz = pos, w = intensity
    vec4 color;    // rgb = color
};

struct SpotLight {
    vec4 position;  // xyz = pos, w = intensity
    vec4 direction; // xyz = dir, w = unused
    vec4 color;     // rgb = color
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

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

vec3 calculateBlinnPhong(vec3 lightDir, vec3 lightColor, vec3 N, vec3 V) {
    vec3 L = normalize(lightDir);
    vec3 H = normalize(L + V);
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), model.shininess);
    
    vec3 diffuse = diff * lightColor * model.albedo.rgb;
    vec3 specular = spec * lightColor * model.specular.rgb;
    return diffuse + specular;
}

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(scene.camera_position.xyz - fragWorldPos);
    
    // Ambient
    vec3 color = scene.ambient_light.rgb * model.albedo.rgb;
    
    // Directional Light
    vec3 dirLightDir = normalize(-scene.dir_light_direction.xyz);
    color += calculateBlinnPhong(dirLightDir, scene.dir_light_color.rgb * scene.dir_light_color.w, N, V);
    
    // Point Lights
    for (int i = 0; i < scene.point_light_count; ++i) {
        vec3 lightVec = point_lights[i].position.xyz - fragWorldPos;
        float distance = length(lightVec);
        vec3 L = lightVec / distance;
        
        // Затухание по закону обратных квадратов
        float attenuation = 1.0 / (distance * distance + 0.01); 
        vec3 lightColor = point_lights[i].color.rgb * point_lights[i].position.w * attenuation;
        
        color += calculateBlinnPhong(L, lightColor, N, V);
    }
    
    // Spot Lights
    for (int i = 0; i < scene.spot_light_count; ++i) {
        vec3 lightVec = spot_lights[i].position.xyz - fragWorldPos;
        float distance = length(lightVec);
        vec3 L = lightVec / distance;
        
        float theta = dot(normalize(-L), normalize(spot_lights[i].direction.xyz));
        float epsilon = cos(radians(spot_lights[i].inner_angle)) - cos(radians(spot_lights[i].outer_angle));
        float spotIntensity = clamp((theta - cos(radians(spot_lights[i].outer_angle))) / epsilon, 0.0, 1.0);
        
        if (spotIntensity > 0.0) {
            float attenuation = 1.0 / (distance * distance + 0.01);
            vec3 lightColor = spot_lights[i].color.rgb * spot_lights[i].position.w * attenuation * spotIntensity;
            color += calculateBlinnPhong(L, lightColor, N, V);
        }
    }
    
    outColor = vec4(color, 1.0);
}