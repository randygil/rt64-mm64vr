//
// RT64
//
// Computes the smooth normal of the vertices of a draw call without lighting from the faces around them, welding the
// vertices with the same position. Faces that are at a sharper angle than the crease are left out, so hard edges stay
// hard. The normals follow the winding of the triangles.
//
// Every triangle is compared with every other one of the draw call, so the triangles are walked in tiles that the
// group loads once into shared memory with their face normals, instead of every thread reading them all from memory.
// All the draw calls of a frame are done in a single dispatch: each group reads its range of indices, the first of its
// triangles and the crease from a table (one entry per group).
//

#define GROUP_SIZE 64

struct RSPSmoothNormalCB {
    uint groupCount;
    uint padding0;
    uint padding1;
};

[[vk::push_constant]] ConstantBuffer<RSPSmoothNormalCB> gConstants : register(b0);
StructuredBuffer<float4> srcWorldPos : register(t1);
StructuredBuffer<uint> srcCol : register(t2);
StructuredBuffer<uint> srcFaceIndices : register(t3);
RWStructuredBuffer<float4> dstWorldNorm : register(u4);

// x first index of the range, y index count of the range, z first triangle of the group in the range, w crease cosine.
StructuredBuffer<uint4> srcGroups : register(t5);

groupshared float3 gTilePositions[GROUP_SIZE * 3];
groupshared float4 gTileNormals[GROUP_SIZE];

// Normal of a face (w = 1 if the face isn't degenerate).
float4 faceNormal(float3 a, float3 b, float3 c) {
    const float3 n = cross(b - a, c - a);
    const float nLength = length(n);
    return (nLength > 1e-8f) ? float4(n / nLength, 1.0f) : float4(0.0f, 0.0f, 0.0f, 0.0f);
}

[numthreads(GROUP_SIZE, 1, 1)]
void CSMain(uint3 groupId : SV_GroupID, uint localIndex : SV_GroupIndex) {
    const uint4 group = srcGroups[groupId.x];
    const uint indexStart = group.x;
    const uint triangleCount = group.y / 3;
    const float creaseCosine = asfloat(group.w);
    const uint triangleIndex = group.z + localIndex;
    const bool ownTriangle = (triangleIndex < triangleCount);
    uint vertexIndices[3] = { 0, 0, 0 };
    float3 positions[3] = { float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f) };
    float4 ownNormal = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (ownTriangle) {
        const uint baseIndex = indexStart + triangleIndex * 3;
        for (uint k = 0; k < 3; k++) {
            vertexIndices[k] = srcFaceIndices[baseIndex + k];
            positions[k] = srcWorldPos[vertexIndices[k]].xyz;
        }

        ownNormal = faceNormal(positions[0], positions[1], positions[2]);
    }

    const float PosDistSqr = 1.0f;
    float3 accumulated[3] = { float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f) };
    for (uint tileStart = 0; tileStart < triangleCount; tileStart += GROUP_SIZE) {
        // Every thread of the group loads one triangle of the tile.
        const uint tileTriangle = tileStart + localIndex;
        if (tileTriangle < triangleCount) {
            const uint baseIndex = indexStart + tileTriangle * 3;
            const float3 a = srcWorldPos[srcFaceIndices[baseIndex + 0]].xyz;
            const float3 b = srcWorldPos[srcFaceIndices[baseIndex + 1]].xyz;
            const float3 c = srcWorldPos[srcFaceIndices[baseIndex + 2]].xyz;
            gTilePositions[localIndex * 3 + 0] = a;
            gTilePositions[localIndex * 3 + 1] = b;
            gTilePositions[localIndex * 3 + 2] = c;
            gTileNormals[localIndex] = faceNormal(a, b, c);
        }
        else {
            gTileNormals[localIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        GroupMemoryBarrierWithGroupSync();

        if (ownTriangle && (ownNormal.w > 0.0f)) {
            const uint tileCount = min(uint(GROUP_SIZE), triangleCount - tileStart);
            for (uint t = 0; t < tileCount; t++) {
                const float4 n = gTileNormals[t];
                if ((n.w <= 0.0f) || (dot(n.xyz, ownNormal.xyz) < creaseCosine)) {
                    continue;
                }

                // Each face counts once for a vertex if any of its corners is at the same position.
                for (uint k = 0; k < 3; k++) {
                    for (uint j = 0; j < 3; j++) {
                        const float3 posDelta = gTilePositions[t * 3 + j] - positions[k];
                        if (dot(posDelta, posDelta) <= PosDistSqr) {
                            accumulated[k] += n.xyz;
                            break;
                        }
                    }
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (ownTriangle && (ownNormal.w > 0.0f)) {
        // W marks the normal as smoothed, as the vertices without lighting don't have normals otherwise.
        for (uint k = 0; k < 3; k++) {
            const float3 n = (dot(accumulated[k], accumulated[k]) > 1e-12f) ? normalize(accumulated[k]) : ownNormal.xyz;
            dstWorldNorm[vertexIndices[k]] = float4(n, 2.0f);
        }
    }
}
