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

layout(set = 2, binding = 0) uniform sampler2D albedo_texture;
layout(set = 2, binding = 1) uniform sampler2D specular_texture;
layout(set = 2, binding = 2) uniform sampler2D emissive_texture;

layout(push_constant) uniform PushConsts {
    int enable_nontrivial;
    float time;
} pc;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

vec3 calculateBlinnPhong(vec3 lightDir, vec3 lightColor, vec3 N, vec3 V, vec3 albedoColor, vec3 specularColor, float shininess) {
    vec3 L = normalize(lightDir);
    vec3 H = normalize(L + V);
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), shininess);
    
    vec3 diffuse = diff * lightColor * albedoColor;
    vec3 specular = spec * lightColor * specularColor;
    return diffuse + specular;
}

void main() {
    vec2 uv = fragUV;
    
    // Нетривиальное сэмплирование
    vec4 albedo = texture(albedo_texture, uv);
    // --- Начало нетривиального сэмплирования ---
    if (pc.enable_nontrivial != 0) {
        float t = pc.time;
        
        // 1. Эффект волновой ряби (Ripple)
        vec2 center = vec2(0.5);
        float dist = length(uv - center);
        // Искажаем координаты круговой волной
        vec2 rippled_uv = uv + sin(dist * 30.0 - t * 5.0) * 0.03;
        
        // 2. Эффект пикселизации (Pixelation)
        // Размер пикселя пульсирует от 20 до 80 (чем больше число, тем крупнее пиксели)
        float pixel_size = mix(20.0, 80.0, (sin(t) * 0.5 + 0.5));
        
        // Округляем координаты до сетки
        vec2 pixelated_uv = floor(rippled_uv * pixel_size) / pixel_size;
        
        // Сэмплируем текстуру с учетом обоих эффектов
        albedo = texture(albedo_texture, pixelated_uv);
        
        // Легкая пульсация цвета (чтобы не было резких вспышек в черный)
        albedo.rgb *= (0.8 + 0.2 * sin(t * 2.0)); 
    }
    // --- Конец нетривиального сэмплирования ---
    
    vec4 specularTex = texture(specular_texture, uv);
    vec4 emissiveTex = texture(emissive_texture, uv);
    
    vec3 albedoColor = albedo.rgb * model.albedo.rgb;
    vec3 specularColor = specularTex.rgb * model.specular.rgb;
    float shininess = model.shininess;
    
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(scene.camera_position.xyz - fragWorldPos);
    
    vec3 color = scene.ambient_light.rgb * albedoColor;
    
    vec3 dirLightDir = normalize(-scene.dir_light_direction.xyz);
    color += calculateBlinnPhong(dirLightDir, scene.dir_light_color.rgb * scene.dir_light_color.w, N, V, albedoColor, specularColor, shininess);
    
    for (int i = 0; i < scene.point_light_count; ++i) {
        vec3 lightVec = point_lights[i].position.xyz - fragWorldPos;
        float distance = length(lightVec);
        vec3 L = lightVec / distance;
        float attenuation = 1.0 / (distance * distance + 0.01);
        vec3 lightColor = point_lights[i].color.rgb * point_lights[i].position.w * attenuation;
        color += calculateBlinnPhong(L, lightColor, N, V, albedoColor, specularColor, shininess);
    }
    
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
            color += calculateBlinnPhong(L, lightColor, N, V, albedoColor, specularColor, shininess);
        }
    }
    
    color += emissiveTex.rgb;
    outColor = vec4(color, 1.0);
}