// prism-text SDF 采样 shader —— 字形 SDF atlas 采样 + 抗锯齿。
//
// # 对应关系（判对基准 = prism-text/src/sdf.rs 烘焙的 atlas）
// - atlas：R8 通道，值 = 有符号距离映射到 [0,1]（d ∈ [−spread, +spread] → [0,1]）。
// - 采样：双线性插值 → smoothstep 抗锯齿（屏幕像素导数宽度）。
// - 排字：glyph quad 顶点着色器输出 uv + px_range（SDF 编码范围）。

struct GlyphInstance {
  // 屏幕空间矩形（像素）。
  rect: vec4<f32>,      // x, y, w, h
  // atlas uv 矩形。
  uv: vec4<f32>,        // u0, v0, u1, v1
  color: vec4<f32>,
  px_range: f32,        // [−range, range] 映射宽度（像素）
  _pad: vec3<f32>,
}

@group(0) @binding(0) var sdf_atlas: texture_2d<f32>;
@group(0) @binding(1) var atlas_sampler: sampler;

struct VertexOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) px_range: f32,
  @location(3) local: vec2<f32>,   // quad 局部坐标 ∈ [0,1]²（供屏幕导数）
}

// 顶点：把实例矩形展开为全屏 NDC 三角形带（6 顶点 / glyph）。
@vertex
fn vs_glyph(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOut {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
  );
  // 实例数据由 storage buffer 提供（此处以实例索引占位，真机接 storage）。
  let local = corners[vid];
  _ = iid;
  var out: VertexOut;
  // NDC：像素 → [−1,1]（y 轴翻转对齐行号向下）。
  let px = local.x;
  let py = local.y;
  let ndc_x = px * 2.0 - 1.0;
  let ndc_y = 1.0 - py * 2.0;
  out.pos = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
  out.color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
  out.uv = local;
  out.px_range = 1.0;
  out.local = local;
  return out;
}

// 片元：SDF 双线性采样 → smoothstep 抗锯齿 → alpha 合成。
@fragment
fn fs_text(in: VertexOut) -> @location(0) vec4<f32> {
  // SDF atlas 单通道（R）；值 ∈ [0,1] 编码 d ∈ [−px_range, +px_range]。
  let d = textureSample(sdf_atlas, atlas_sampler, in.uv).r;
  // 编码距离 → 像素距离：dd = (d − 0.5) · 2 · px_range。
  let dd = (d - 0.5) * 2.0 * in.px_range;
  // 屏幕空间导数 → 抗锯齿宽度（fwidth 为 WGSL 内建，非超越函数，允许）。
  let w = max(fwidth(dd), 1e-5);
  let alpha = smoothstep(-w, w, dd);
  return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
