#version 330
layout(location = 0) in vec3 position;
// layout(location = 1) in vec3 vert_color;
layout(location = 1) in vec3 texcoord;
layout(location = 2) in vec3 normal;

// uniform mat4 transform;

out vec4 worldPosition;
// out vec3 vertColor;
out vec3 OutNormal;
out vec3 OutCoord;

void main()
{
    // vec4 tempPosition = transform * vec4(0.0f) + vec4(position, 1.0f);
    // worldPosition = tempPosition;
    // gl_Position = tempPosition + vec4((texcoord), 0.0f); // world space
    gl_Position = vec4(position, 1.0f);
    worldPosition = vec4(position, 1.0f);

    // vertColor = vert_color;
    //gl_Position = vec4((texcoord.xy*2.0 - 1.0), 0.0, 1.0f); // UV space
    //OutNormal = (transform * vec4(normal, 1.0)).xyz;
    // OutNormal = vec3(texcoord.xy, 0.0) + normal * 0.000000001;
    OutNormal = normal;
    OutCoord  = texcoord;
}