fn main() {
    for n in ["a", "b", "c"] {
        let c = std::fs::read(format!("/tmp/zt/{n}.z")).unwrap();
        let raw = std::fs::read(format!("/tmp/zt/{n}.raw")).unwrap();
        match prism_io::exr::zlib_decompress(&c) {
            Ok(g) => println!("{n}: ok len={} want={} eq={}", g.len(), raw.len(), g == raw),
            Err(e) => println!("{n}: ERR {e:?}"),
        }
    }
}
