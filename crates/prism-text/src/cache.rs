//! 确定性字形位图缓存（任务书 B5：缓存命中前后逐字节一致）。
//!
//! 设计：
//! - **纯加速**：缓存命中与未命中必须产出**同一**位图——渲染是
//!   (gid, size, flags) 的纯函数，缓存只是 memo（测试锁定逐字节一致）；
//! - **键**：`gid + size_bits + flags`。size 以 `f32::to_bits()` 入键——
//!   位级同一性（0.1 与 0.10000000147 视为不同键，宁可重渲不串键）；
//! - **容器**：`BTreeMap`——迭代序确定（哈希容器迭代序不确定，禁用）。
//!
//! flags 位分配（当前仅位 0）：`0` = 直立；预留 `1` = 顺时针 90°
//! （竖排转向渲染时由调用方先 `rotate_cw` 再入缓存亦可直接查）。

use std::collections::BTreeMap;

use crate::raster::{render_glyph, GlyphBitmap};
use crate::ttf::Font;

/// 缓存键（位级精确同一性）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GlyphKey {
    /// 字形 id。
    pub gid: u16,
    /// 字号的 f32 位模式（`size.to_bits()`）。
    pub size_bits: u32,
    /// 渲染旗标（位 0：0=直立，1=顺时针 90°；其余预留须为 0）。
    pub flags: u32,
}

/// 字形位图缓存（进程内 memo；BTreeMap 迭代序确定）。
#[derive(Debug, Default, Clone)]
pub struct GlyphCache {
    map: BTreeMap<GlyphKey, GlyphBitmap>,
}

impl GlyphCache {
    /// 构造空缓存。
    pub fn new() -> Self {
        Self::default()
    }

    /// 缓存条目数。
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// 取字形位图：命中返回缓存引用；未命中渲染后入缓存。
    ///
    /// `flags & 1 != 0` 时先渲染直立位图再顺时针旋转 90° 入缓存——
    /// 同键命中返回同一旋转结果（确定性不变式）。
    pub fn get_or_render(
        &mut self,
        font: &Font,
        gid: u16,
        size: f32,
        flags: u32,
    ) -> &GlyphBitmap {
        let key = GlyphKey { gid, size_bits: size.to_bits(), flags };
        self.map.entry(key).or_insert_with(|| {
            let bmp = render_glyph(font, gid, size);
            if flags & 1 != 0 {
                bmp.rotate_cw()
            } else {
                bmp
            }
        })
    }

    /// 只读查看某键是否已缓存（测试与诊断用）。
    pub fn contains(&self, gid: u16, size: f32, flags: u32) -> bool {
        self.map.contains_key(&GlyphKey { gid, size_bits: size.to_bits(), flags })
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::font_regular;

    #[test]
    fn cold_and_warm_hits_are_byte_identical() {
        let f = font_regular();
        let gid = f.glyph_index('景').expect("景 在子集内");
        let mut cache = GlyphCache::new();
        assert!(cache.is_empty());
        // 冷渲染。
        let cold = cache.get_or_render(&f, gid, 32.0, 0).clone();
        assert_eq!(cache.len(), 1);
        // 热命中：与冷渲染逐字节一致（同对象同一性 + 值相等双断言）。
        let warm = cache.get_or_render(&f, gid, 32.0, 0);
        assert_eq!(*warm, cold, "缓存命中必须与冷渲染逐字节一致");
        assert_eq!(warm.alpha, cold.alpha);
        assert_eq!(cache.len(), 1, "命中不得新增条目");
        // 位级同一性：0.1 与其位翻转邻居是不同键（宁可重渲不串键）。
        cache.get_or_render(&f, gid, 0.1, 0);
        assert!(cache.contains(gid, 0.1_f32, 0));
        assert!(!cache.contains(gid, f32::from_bits(0.1_f32.to_bits() ^ 1), 0));
        assert_eq!(cache.len(), 2, "32px 与 0.1px 两个条目");
    }

    #[test]
    fn distinct_sizes_are_distinct_entries() {
        let f = font_regular();
        let gid = f.glyph_index('形').expect("形 在子集内");
        let mut cache = GlyphCache::new();
        let a = cache.get_or_render(&f, gid, 24.0, 0).clone();
        let b = cache.get_or_render(&f, gid, 32.0, 0).clone();
        assert_eq!(cache.len(), 2, "不同字号独立缓存");
        assert_ne!(a.width.max(a.height), 0);
        // 24px 版图更小或相等（同字形字号单调）。
        assert!(a.width <= b.width && a.height <= b.height);
    }

    #[test]
    fn rotated_flag_key_is_independent() {
        let f = font_regular();
        let gid = f.glyph_index('标').expect("标 在子集内");
        let mut cache = GlyphCache::new();
        let upright = cache.get_or_render(&f, gid, 24.0, 0).clone();
        let rotated = cache.get_or_render(&f, gid, 24.0, 1).clone();
        assert_eq!(cache.len(), 2, "flags 入键");
        assert_eq!(
            (rotated.width, rotated.height),
            (upright.height, upright.width),
            "转向位图宽高互换"
        );
        // 与直接 rotate_cw 一致（命中路径同构）。
        assert_eq!(rotated, upright.rotate_cw());
    }
}
