#version 330
in vec4 worldPosition;
// in vec3 vertColor;
in vec3 OutNormal;
in vec3 OutCoord;

out vec4 OutColor;

uniform sampler2D samplerTex;

void main()
{          
    // vec3 temp = worldPosition.xyz;
    // temp = (temp + 1.0) * 0.5;

    vec2 uv = OutCoord.xy;

    // uv = uv * 0.5;
    uv.y = - uv.y; // flip texture
    
    // vec3 camVec = vec3(0.0, 0.0, 1.0);
    // float Vis = dot(camVec, OutNormal);// * 2.0 - 1.0;

    //if(Vis>0.0) OutColor = texture(samplerTex, uv);
    float zero = 0.000000001;
    OutColor = texture(samplerTex, uv) + vec4(OutNormal, 0.0f) + worldPosition * zero;
    // OutColor = vec4(vertColor, 1.0f);
}