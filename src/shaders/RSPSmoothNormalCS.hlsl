//
// RT64
//
// Computes the smooth normal of the vertices of a draw call without lighting from the faces around them, welding the
// vertices with the same position. Faces that are at a sharper angle than the crease are left out, so hard edges stay
// hard. The normals follow the winding of the triangles.
//

#define GROUP_SIZE 64

struct RSPSmoothNormalCB {
    uint indexStart;
    uint indexCount;
    float creaseCosine;
};

[[vk::push_constant]] ConstantBuffer<RSPSmoothNormalCB> gConstants : register(b0);
StructuredBuffer<float4> srcWorldPos : register(t1);
StructuredBuffer<uint> srcCol : register(t2);
StructuredBuffer<uint> srcFaceIndices : register(t3);
RWStructuredBuffer<float4> dstWorldNorm : register(u4);

float3 faceNormal(uint triangleIndex, out bool valid) {
    const uint baseIndex = gConstants.indexStart + triangleIndex * 3;
    const float3 a = srcWorldPos[srcFaceIndices[baseIndex + 0]].xyz;
    const float3 b = srcWorldPos[srcFaceIndices[baseIndex + 1]].xyz;
    const float3 c = srcWorldPos[srcFaceIndices[baseIndex + 2]].xyz;
    const float3 n = cross(b - a, c - a);
    const float nLength = length(n);
    valid = (nLength > 1e-8f);
    return valid ? (n / nLength) : float3(0.0f, 0.0f, 0.0f);
}

float3 computeSmoothNormal(uint vertexIndex, float3 ownNormal) {
    const float PosDistSqr = 1.0f;
    const float3 vertexPos = srcWorldPos[vertexIndex].xyz;
    float3 vertexNorm = float3(0.0f, 0.0f, 0.0f);
    const uint triangleCount = gConstants.indexCount / 3;
    for (uint t = 0; t < triangleCount; t++) {
        for (uint j = 0; j < 3; j++) {
            const uint cmpIndex = srcFaceIndices[gConstants.indexStart + t * 3 + j];
            const float3 posDelta = srcWorldPos[cmpIndex].xyz - vertexPos;
            if (dot(posDelta, posDelta) <= PosDistSqr) {
                bool valid;
                const float3 n = faceNormal(t, valid);
                if (valid && (dot(n, ownNormal) >= gConstants.creaseCosine)) {
                    vertexNorm += n;
                }

                break;
            }
        }
    }

    return (dot(vertexNorm, vertexNorm) > 1e-12f) ? normalize(vertexNorm) : ownNormal;
}

[numthreads(GROUP_SIZE, 1, 1)]
void CSMain(uint triangleIndex : SV_DispatchThreadID) {
    bool valid = ((triangleIndex * 3) < gConstants.indexCount);
    uint v0 = 0;
    uint v1 = 0;
    uint v2 = 0;
    float3 n0 = float3(0.0f, 0.0f, 0.0f);
    float3 n1 = float3(0.0f, 0.0f, 0.0f);
    float3 n2 = float3(0.0f, 0.0f, 0.0f);
    if (valid) {
        const float3 ownNormal = faceNormal(triangleIndex, valid);
        if (valid) {
            const uint baseIndex = gConstants.indexStart + triangleIndex * 3;
            v0 = srcFaceIndices[baseIndex + 0];
            v1 = srcFaceIndices[baseIndex + 1];
            v2 = srcFaceIndices[baseIndex + 2];

            n0 = computeSmoothNormal(v0, ownNormal);
            n1 = computeSmoothNormal(v1, ownNormal);
            n2 = computeSmoothNormal(v2, ownNormal);
        }
    }

    AllMemoryBarrierWithGroupSync();
    if (valid) {
        // W marks the normal as smoothed, as the vertices without lighting don't have normals otherwise.
        dstWorldNorm[v0] = float4(n0, 2.0f);
        dstWorldNorm[v1] = float4(n1, 2.0f);
        dstWorldNorm[v2] = float4(n2, 2.0f);
    }
}
