//! 临时工具：读取官方 DWA 文件，打印解码结果。验证后删除。
use prism_io::exr;
use std::fs;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: zz_probe <file.exr> [...]");
        std::process::exit(2);
    }
    args.sort();
    for p in args {
        let data = match fs::read(&p) {
            Ok(d) => d,
            Err(e) => { println!("{p}: READ FILE FAIL {e}"); continue; }
        };
        match exr::read_image(&data) {
            Ok(img) => {
                let mut s = String::new();
                for pl in &img.planes {
                    s.push_str(&format!("[{} {:?} n={} ", pl.name, pl.pixel_type, pl.values.len()));
                    for v in pl.values.iter().take(4) {
                        s.push_str(&format!("{v:e} "));
                    }
                    s.push_str("] ");
                }
                println!("{p}: OK {}", img.planes.len());
                println!("   {s}");
            }
            Err(e) => println!("{p}: DECODE FAIL {e:?}"),
        }
    }
}