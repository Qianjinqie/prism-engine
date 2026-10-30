//! `prism-ffi-hwenc` 的 VA-API **绑定就绪层**外部钉子（`--features vaapi` 才编译）。
//!
//! 守的是三件事，缺一条本项即不成立：
//!
//! 1. **第二档要说实话**：`libva.so.2` 能 `dlopen` 就必须把 18 个核心符号全部
//!    解析进函数表；不能则必须给**路径 + errno + dlerror** 的结构化理由。
//! 2. **永不 panic、永不静默**：失败路径走的是真 `dlopen`（不存在的库名/路径），
//!    不是 mock——拿到的就是内核/loader 的真错误。
//! 3. **永不越级**：任何探测理由都不得声称第三/第四档；无 `/dev/dri` 时结论必须
//!    是 `Unsupported` 且证据串非空。
//!
//! 本文件不含 `#[ignore]`：它是可自动执行的（沙箱无 `/dev/dri`、库有/无两种宿主
//! 都能跑通）。
#![cfg(feature = "vaapi")]

use prism_ffi_hwenc::{
    any_available, probe_all, vaapi_bind_from, vaapi_bind_report, vaapi_tier, HwBackend,
    ProbeStatus, VaapiBindError, VaapiBindReport, VaapiOpenCause, DRI_DEV_DIR, LIBVA_SONAME,
    Maturity, REQUIRED_SYMBOLS,
};

/// 一个**确定不存在**的库名（dlopen 必然失败，且失败原因可预期为 ENOENT）。
const ABSENT_LIBRARY: &str = "libva-binding-probe-does-not-exist.so.2";

#[test]
fn absent_library_yields_structured_error_with_path_and_errno() {
    // 验收 ②：dlopen 失败 ⇒ 结构化 Unsupported 证据（路径 + errno），不 panic。
    let err: VaapiBindError = match vaapi_bind_from(ABSENT_LIBRARY) {
        Ok(_) => panic!("{ABSENT_LIBRARY} 不该存在；dlopen 失败路径未被走到"),
        Err(e) => e,
    };
    let evidence: String = err.evidence();
    assert!(!evidence.is_empty(), "证据串不得为空");
    assert!(evidence.contains(ABSENT_LIBRARY), "证据必须带库名/路径：{evidence}");
    assert!(evidence.contains("errno="), "证据必须带 errno：{evidence}");
    assert!(evidence.contains("dlerror="), "证据必须带 dlerror 原文位：{evidence}");
    assert!(err.is_library_not_loadable(), "库不存在属于打不开，不是符号缺");
    assert_eq!(err.library(), ABSENT_LIBRARY, "库名必须原样回显");
    // Display 与 evidence() 必须同源（日志与断言看到同一串）。
    assert_eq!(err.to_string(), evidence, "Display 必须与 evidence() 同源");
    // ENOENT 面：glibc 的 loader **不保证**设 errno（实测为 0，已归一化为 None），
    // 故 errno 只能"未设置或 2"，而真正的证据是 dlerror 原文与成因分类。
    assert!(evidence.contains("cause=not-found"), "证据必须给出成因分类：{evidence}");
    assert_eq!(err.cause(), VaapiOpenCause::NotFound);
    match err {
        VaapiBindError::LibraryNotLoadable { errno, errno_text, detail, .. } => {
            assert_ne!(errno, Some(0), "errno=0 必须归一化为 None，不许当证据");
            assert!(errno.is_none() || errno == Some(2), "errno 只能是未设置或 ENOENT");
            if errno.is_some() {
                assert!(!errno_text.is_empty(), "有 errno 就必须有文本");
            }
            assert!(
                detail.contains("No such file or directory")
                    || detail.contains("cannot open shared object file"),
                "dlerror 原文必须说清打不开的原因：{detail:?}"
            );
        }
        VaapiBindError::SymbolMissing { symbol, .. } => {
            panic!("不存在的库不该走到符号解析（{symbol}）")
        }
    }
}

#[test]
fn absent_library_absolute_path_is_echoed_and_never_panics() {
    // 绝对路径同样原样回显（运维可拿去对照 LD_LIBRARY_PATH / ldconfig）。
    let path: &str = "/nonexistent/vaapi-binding-probe/libva.so.2";
    match vaapi_bind_from(path) {
        Ok(_) => panic!("{path} 不该存在"),
        Err(err) => {
            assert_eq!(err.library(), path);
            assert!(err.evidence().contains(path));
        }
    }
}

#[test]
fn default_soname_probe_is_structural_and_repeatable() {
    // 绑定探测：要么就绪（18/18 符号），要么给出非空证据；两跑逐字段相同。
    let a: &VaapiBindReport = vaapi_bind_report();
    let b: &VaapiBindReport = vaapi_bind_report();
    assert_eq!(a, b, "同一进程内两次探测必须逐字段相同");
    assert_eq!(a.library, LIBVA_SONAME, "默认探测对象必须是 libva.so.2");
    assert!(!a.summary().is_empty(), "摘要不得为空（失败也要说清理由）");

    if a.is_ready() {
        // 宿主有 libva：绑定就绪档成立，符号面必须齐。
        assert_eq!(a.resolved.len(), REQUIRED_SYMBOLS.len(), "就绪必须 18/18");
        assert!(a.error.is_none());
        assert!(a.summary().contains(LIBVA_SONAME));
        assert_eq!(vaapi_tier(), Maturity::BindingReady);
    } else {
        // 宿主无 libva：必须是"带证据的未就绪"，不许静默。
        assert!(a.resolved.is_empty(), "未就绪时不得有已解析符号");
        let err: &VaapiBindError = a.error.as_ref().expect("未就绪必须有结构化理由");
        assert!(!err.evidence().is_empty());
        assert!(a.summary().contains("errno="), "摘要必须带 errno 证据：{}", a.summary());
        assert_eq!(vaapi_tier(), Maturity::ProbeChain);
    }
}

#[test]
fn tier_never_exceeds_binding_ready() {
    // 验收 ③：四档分述不混用——本 crate 的上限就是第二档。
    let tier: Maturity = vaapi_tier();
    assert!(tier.is_at_most(Maturity::BindingReady), "档位越级：{tier}");
    assert!(tier.level() <= 2, "level 也必须是 1 或 2，实得 {}", tier.level());
    assert_ne!(tier, Maturity::SessionEstablished);
    assert_ne!(tier, Maturity::DeviceEncoded);
}

#[test]
fn probe_reason_carries_real_environment_evidence_and_stays_unsupported() {
    // 验收 ④（沙箱即无 /dev/dri 环境）：结论 Unsupported + 证据串非空。
    let vaapi_probes: Vec<_> = probe_all()
        .into_iter()
        .filter(|p| p.backend == HwBackend::Vaapi)
        .collect();
    assert!(!vaapi_probes.is_empty(), "vaapi feature 开启时必须有 VA-API 探测行");
    for probe in &vaapi_probes {
        assert_eq!(probe.status, ProbeStatus::Unsupported, "{} 不得声称可用", probe.codec);
        assert!(!probe.reason.is_empty(), "理由不得为空");
        assert!(
            probe.reason.contains(DRI_DEV_DIR),
            "理由必须带 /dev/dri 证据：{}",
            probe.reason
        );
        // 绑定档结论必须出现在理由里（第二档是上限，且要说清楚）。
        assert!(
            probe.reason.contains(Maturity::BindingReady.as_str())
                || probe.reason.contains(Maturity::ProbeChain.as_str()),
            "理由必须写明当前档位：{}",
            probe.reason
        );
    }
    assert!(!any_available(), "无会话即无可用编码器");

    // 沙箱专属断言：无 /dev/dri 时必须点明"缺渲染节点"这条真实证据。
    if !std::path::Path::new(DRI_DEV_DIR).exists() {
        let reason: &str = vaapi_probes[0].reason.as_str();
        assert!(
            reason.contains("渲染节点"),
            "无 {DRI_DEV_DIR} 时理由必须点明渲染节点缺席：{reason}"
        );
        assert!(reason.len() > 20, "证据串太短，疑似占位：{reason}");
    }
}

#[test]
fn required_symbols_cover_the_named_core_api() {
    // 符号面钉子：任务书点名的 14 个核心符号必须逐个在册。
    for must in [
        "vaInitialize",
        "vaTerminate",
        "vaQueryConfigProfiles",
        "vaCreateConfig",
        "vaDestroyConfig",
        "vaCreateContext",
        "vaDestroyContext",
        "vaCreateSurfaces",
        "vaDestroySurfaces",
        "vaBeginPicture",
        "vaRenderPicture",
        "vaEndPicture",
        "vaMapBuffer",
        "vaUnmapBuffer",
    ] {
        assert!(REQUIRED_SYMBOLS.contains(&must), "核心符号 {must} 缺失");
    }
    assert_eq!(REQUIRED_SYMBOLS.len(), 18, "核心符号数应固定为 18");
    let mut uniq: Vec<&str> = REQUIRED_SYMBOLS.to_vec();
    uniq.sort_unstable();
    uniq.dedup();
    assert_eq!(uniq.len(), REQUIRED_SYMBOLS.len(), "符号名不得重复");
}
