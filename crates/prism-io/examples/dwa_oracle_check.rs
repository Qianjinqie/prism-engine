//! DWA 解码器对真实 OpenEXR 文件的验证 harness（不入库测试，仅开发期使用）。
use prism_io::exr;
use std::fs;

fn f32_from_bits(b: u32) -> f32 { f32::from_bits(b) }

fn check(exr_path: &str, expect_path: &str) {
    let data = fs::read(exr_path).expect("exr");
    let exp = fs::read_to_string(expect_path).expect("expect");
    let mut lines = exp.lines();
    let dims: Vec<u32> = lines.next().unwrap().split_whitespace().map(|s| s.parse().unwrap()).collect();
    let (w, h) = (dims[0], dims[1]);
    let img = match exr::read_image(&data) {
        Ok(i) => i,
        Err(e) => { println!("{exr_path}: READ FAIL {e:?}"); return; }
    };
    println!("{exr_path}: {}x{} channels={:?}", w, h,
        img.planes.iter().map(|p| (p.name.clone(), p.pixel_type)).collect::<Vec<_>>());
    let mut maxdiff = 0.0f32;
    let mut nmis = 0usize;
    let mut total = 0usize;
    for (i, line) in lines.enumerate() {
        let want: Vec<u32> = line.split_whitespace().map(|s| u32::from_str_radix(s, 16).unwrap()).collect();
        for (c, wbits) in want.iter().enumerate() {
            let name = ["R", "G", "B", "Z"][c];
            let plane = img.plane(name).unwrap_or_else(|| panic!("no plane {name}"));
            let got = plane.values[i];
            let wv = f32_from_bits(*wbits);
            let d = (got - wv).abs();
            let rel = d / wv.abs().max(1e-6);
            if rel > maxdiff { maxdiff = rel; }
            if got.to_bits() != *wbits {
                nmis += 1;
                if nmis <= 6 {
                    println!("  [{i},{name}] got={:08x}({got}) want={wbits:08x}({wv})", got.to_bits());
                }
            }
            total += 1;
        }
    }
    println!("  total={total} mismatch={nmis} max_rel_diff={maxdiff:.6}");
}

fn main() {
    for (a, b) in [
        ("/tmp/oracle/dwaa_a.exr", "/tmp/oracle/dwaa_a.txt"),
        ("/tmp/oracle/dwab_a.exr", "/tmp/oracle/dwab_a.txt"),
        ("/tmp/oracle/dwab_b.exr", "/tmp/oracle/dwab_b.txt"),
        ("/tmp/oracle/dwab_c.exr", "/tmp/oracle/dwab_c.txt"),
        ("/tmp/oracle/zip_real.exr", "/tmp/oracle/zip_real.txt"),
        ("/tmp/oracle/zips_real.exr", "/tmp/oracle/zips.txt"),
    ] { check(a, b); }
}
