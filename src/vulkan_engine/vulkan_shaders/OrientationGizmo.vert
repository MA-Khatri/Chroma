#version 450

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;
layout(location = 2) in vec3 a_Color;
layout(location = 3) in vec2 a_TexCoord;

layout(location = 0) out vec3 fragColor;

layout(set = 0, binding = 0) uniform SceneUBO {
  mat4 view;
  mat4 proj;
  mat4 viewProj;
  vec3 cameraPosition;
  vec2 viewportSize;
}
scene;

layout(set = 1, binding = 0) uniform ObjectUBO {
  mat4 model;
  mat4 normal;
}
object;

// --- Gizmo appearance controls ---
// Size of the gizmo, as a fraction of viewport HEIGHT (0..1), measured as
// the distance from its center to the tip of an axis.
const float kGizmoSizePercent = 0.08;
// Padding between the gizmo and the top/right edges of the viewport, also
// as a fraction of viewport HEIGHT (0..1).
const float kGizmoPaddingPercent = 0.03;

void main() {
  // 1. Strip translation from the view matrix so we only keep the camera's
  //    orientation. The gizmo's axes rotate with the camera but never move.
  mat3 viewRotation = mat3(scene.view);
  vec3 rotated = viewRotation * a_Position;

  // 2. Convert screen-percent controls into pixels, relative to viewport
  //    HEIGHT so the gizmo's size tracks vertical resolution only.
  float gizmoSizePixels = kGizmoSizePercent * scene.viewportSize.y;
  float gizmoPaddingPixels = kGizmoPaddingPercent * scene.viewportSize.y;

  // 3. Isotropic pixel offset (keeps the gizmo circular before aspect
  //    correction).
  vec2 pixelOffset = rotated.xy * gizmoSizePixels;

  // 4. Convert to NDC space per-axis to correct for aspect ratio.
  vec2 halfViewport = scene.viewportSize * 0.5;
  vec2 ndcOffset = pixelOffset / halfViewport;

  // 5. NDC-space center, inset from the top-right corner.
  //    Vulkan NDC is normally y-down (+1 = bottom), so top-right is
  //    (+1, -1). Since the pipeline flips the viewport downstream, we
  //    negate y here so the final on-screen result still lands top-right
  //    after that later flip is applied.
  vec2 insetPixels = vec2(gizmoSizePixels + gizmoPaddingPixels);
  vec2 insetNDC = insetPixels / halfViewport;
  vec2 gizmoCenterNDC = vec2(1.0 - insetNDC.x, 1.0 - insetNDC.y);

  vec2 ndcPosition = gizmoCenterNDC + vec2(ndcOffset.x, -ndcOffset.y);

  // 6. Depth: Vulkan's default NDC depth range is [0, 1], not [-1, 1].
  //    rotated.z is roughly in [-1, 1] (unit-scale gizmo geometry), so
  //    bias/scale it into a tiny band centered at a fixed depth rather
  //    than letting negative z go below 0 and get near-plane clipped.
  //    0.5 is an arbitrary "always draw on top of nothing in particular"
  //    depth — adjust if you're depth-testing the gizmo against itself.
  const float kDepthBase = 0.5;
  const float kDepthScale = 0.0001;
  float pseudoDepth = kDepthBase + rotated.z * kDepthScale;

  gl_Position = vec4(ndcPosition, pseudoDepth, 1.0);
  fragColor = a_Color;
}