#version 450
layout(location = 0) in vec3 inPosition;
layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 color;
} pc;
layout(location = 0) out vec4 fragColor;
void main() {
    fragColor = pc.color;
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
}