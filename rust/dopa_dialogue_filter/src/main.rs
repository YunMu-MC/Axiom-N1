use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Verdict {
    accepted: bool,
    reason: &'static str,
}

#[derive(Debug, Clone)]
struct CleanPolicyRuntime {
    min_chars: usize,
    max_chars: usize,
    min_score: f64,
    allowed_licenses: HashSet<String>,
    allowed_languages: HashSet<String>,
    target_categories: HashSet<String>,
}

impl CleanPolicyRuntime {
    fn new(
        min_chars: usize,
        max_chars: usize,
        min_score: f64,
        allowed_licenses: HashSet<String>,
        allowed_languages: HashSet<String>,
        target_categories: HashSet<String>,
    ) -> Self {
        Self {
            min_chars,
            max_chars,
            min_score,
            allowed_licenses,
            allowed_languages,
            target_categories,
        }
    }
}

#[derive(Default)]
struct CleanState {
    fingerprints: HashSet<(u64, usize)>,
    thread_keys: HashSet<String>,
}

struct CleanInputRow {
    index: String,
    source: String,
    source_id: String,
    license: String,
    language: String,
    turns: usize,
    thread_key: String,
    metadata_json: String,
    text: String,
}

fn prefilter_text(
    license: &str,
    language: &str,
    turns: usize,
    text: &str,
    min_chars: usize,
    max_chars: usize,
    allowed_licenses: &HashSet<String>,
    allowed_languages: &HashSet<String>,
) -> Verdict {
    let license = license.trim().to_ascii_lowercase();
    if !allowed_licenses.contains(&license) {
        return reject("license_not_allowed");
    }
    let language = normalize_language(language);
    if !language.is_empty() && !allowed_languages.contains(&language) {
        return reject("language_not_allowed");
    }
    if turns < 2 {
        return reject("too_few_turns");
    }
    if turns > 24 {
        return reject("too_many_turns");
    }
    let char_count = text.chars().count();
    if char_count < min_chars {
        return reject("too_short");
    }
    if char_count > max_chars {
        return reject("too_long");
    }
    let lower = text.to_ascii_lowercase();
    if contains_email(text) || contains_secret(&lower) {
        return reject("pii_or_secret");
    }
    if contains_bad_code_artifact(&lower) {
        return reject("bad_code_artifact");
    }
    if contains_mojibake(text) {
        return reject("mojibake");
    }
    if is_url_only(text) {
        return reject("url_only");
    }
    if bad_unicode_ratio(text) > 0.01 {
        return reject("bad_unicode");
    }
    if contains_unsafe_security(&lower) {
        return reject("unsafe_security");
    }
    if contains_financial_market_advice(&lower) {
        return reject("financial_market_advice");
    }
    if repetition_ratio(&lower) > 0.35 {
        return reject("repetition");
    }
    Verdict {
        accepted: true,
        reason: "accepted",
    }
}

fn filter_reader<R: BufRead, W: Write>(
    reader: R,
    mut writer: W,
    min_chars: usize,
    max_chars: usize,
    allowed_licenses: &HashSet<String>,
    allowed_languages: &HashSet<String>,
) -> Result<(), String> {
    for (line_no, line_result) in reader.lines().enumerate() {
        let line = line_result.map_err(|err| err.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        write_verdict_for_line(
            line_no + 1,
            &line,
            &mut writer,
            min_chars,
            max_chars,
            allowed_licenses,
            allowed_languages,
        )?;
    }
    writer.flush().map_err(|err| err.to_string())
}

fn filter_batch_stream<R: BufRead, W: Write>(
    mut reader: R,
    mut writer: W,
    min_chars: usize,
    max_chars: usize,
    allowed_licenses: &HashSet<String>,
    allowed_languages: &HashSet<String>,
) -> Result<(), String> {
    let mut header = String::new();
    let mut row = String::new();
    let mut line_no = 0usize;
    loop {
        header.clear();
        let header_bytes = reader.read_line(&mut header).map_err(|err| err.to_string())?;
        if header_bytes == 0 {
            break;
        }
        let header = header.trim();
        if header.is_empty() {
            continue;
        }
        let raw_count = header.strip_prefix("batch\t").unwrap_or(header);
        let batch_rows = raw_count
            .parse::<usize>()
            .map_err(|err| format!("invalid batch header '{}': {}", header, err))?;
        let mut output_lines: Vec<String> = Vec::with_capacity(batch_rows);
        for _ in 0..batch_rows {
            row.clear();
            let row_bytes = reader.read_line(&mut row).map_err(|err| err.to_string())?;
            if row_bytes == 0 {
                return Err("unexpected EOF while reading batch rows".to_string());
            }
            line_no += 1;
            let line = row.trim_end_matches(|ch| ch == '\r' || ch == '\n');
            output_lines.push(verdict_line_for_line(
                line_no,
                line,
                min_chars,
                max_chars,
                allowed_licenses,
                allowed_languages,
            )?);
        }
        for output_line in output_lines {
            writer
                .write_all(output_line.as_bytes())
                .map_err(|err| err.to_string())?;
        }
        writer.flush().map_err(|err| err.to_string())?;
    }
    writer.flush().map_err(|err| err.to_string())
}

fn write_verdict_for_line<W: Write>(
    line_no: usize,
    line: &str,
    writer: &mut W,
    min_chars: usize,
    max_chars: usize,
    allowed_licenses: &HashSet<String>,
    allowed_languages: &HashSet<String>,
) -> Result<(), String> {
    let output = verdict_line_for_line(
        line_no,
        line,
        min_chars,
        max_chars,
        allowed_licenses,
        allowed_languages,
    )?;
    writer
        .write_all(output.as_bytes())
        .map_err(|err| err.to_string())
}

fn verdict_line_for_line(
    line_no: usize,
    line: &str,
    min_chars: usize,
    max_chars: usize,
    allowed_licenses: &HashSet<String>,
    allowed_languages: &HashSet<String>,
) -> Result<String, String> {
    let parts: Vec<&str> = line.splitn(5, '\t').collect();
    if parts.len() != 5 {
        return Err(format!("invalid TSV line {}: expected 5 columns", line_no));
    }
    let turns = parts[3].parse::<usize>().map_err(|err| err.to_string())?;
    let verdict = prefilter_text(
        parts[1],
        parts[2],
        turns,
        parts[4],
        min_chars,
        max_chars,
        allowed_licenses,
        allowed_languages,
    );
    let action = if verdict.accepted { "accept" } else { "reject" };
    Ok(format!("{}	{}	{}\n", parts[0], action, verdict.reason))
}

fn clean_batch_stream<R: BufRead, W: Write>(
    mut reader: R,
    mut writer: W,
    policy: &CleanPolicyRuntime,
    state: &mut CleanState,
) -> Result<(), String> {
    let mut header = String::new();
    let mut row = String::new();
    let mut line_no = 0usize;
    loop {
        header.clear();
        let header_bytes = reader.read_line(&mut header).map_err(|err| err.to_string())?;
        if header_bytes == 0 {
            break;
        }
        let header = header.trim();
        if header.is_empty() {
            continue;
        }
        let raw_count = header.strip_prefix("batch\t").unwrap_or(header);
        let batch_rows = raw_count
            .parse::<usize>()
            .map_err(|err| format!("invalid clean batch header '{}': {}", header, err))?;
        let mut output_lines: Vec<String> = Vec::with_capacity(batch_rows);
        for _ in 0..batch_rows {
            row.clear();
            let row_bytes = reader.read_line(&mut row).map_err(|err| err.to_string())?;
            if row_bytes == 0 {
                return Err("unexpected EOF while reading clean batch rows".to_string());
            }
            line_no += 1;
            let line = row.trim_end_matches(|ch| ch == '\r' || ch == '\n');
            output_lines.push(clean_line_for_row(line_no, line, policy, state)?);
        }
        for output_line in output_lines {
            writer
                .write_all(output_line.as_bytes())
                .map_err(|err| err.to_string())?;
        }
        writer.flush().map_err(|err| err.to_string())?;
    }
    writer.flush().map_err(|err| err.to_string())
}

fn clean_line_for_row(
    line_no: usize,
    line: &str,
    policy: &CleanPolicyRuntime,
    state: &mut CleanState,
) -> Result<String, String> {
    let row = parse_clean_input_row(line_no, line)?;
    if let Some(reason) = hard_reject_reason(&row, policy) {
        return Ok(reject_line(&row.index, reason));
    }
    let lower = row.text.to_ascii_lowercase();
    let (score, category, tags) = score_dialogue(&row.text, &lower, row.turns);
    if score < policy.min_score {
        return Ok(reject_line(&row.index, "low_quality"));
    }
    if !policy.target_categories.is_empty() && !policy.target_categories.contains(category) {
        return Ok(reject_line(&row.index, "category_not_target"));
    }
    if !row.thread_key.is_empty() && state.thread_keys.contains(&row.thread_key) {
        return Ok(reject_line(&row.index, "duplicate_thread"));
    }
    let fingerprint = fingerprint_key(&row.text);
    if state.fingerprints.contains(&fingerprint) {
        return Ok(reject_line(&row.index, "duplicate"));
    }
    let language = language_group_for_text(&row.language, &row.text);
    let json_line = build_training_json_line(&row, &language, category, score, &tags);
    let encoded_len = json_line.as_bytes().len();
    state.fingerprints.insert(fingerprint);
    if !row.thread_key.is_empty() {
        state.thread_keys.insert(row.thread_key.clone());
    }
    Ok(format!(
        "{}\taccept\taccepted\t{}\t{}\t{:.4}\t{}\t{}\t{}\t{}",
        row.index,
        language,
        category,
        score,
        tags.join(","),
        encoded_len,
        row.source,
        json_line
    ))
}

fn parse_clean_input_row(line_no: usize, line: &str) -> Result<CleanInputRow, String> {
    let parts: Vec<&str> = line.splitn(9, '\t').collect();
    if parts.len() != 9 {
        return Err(format!("invalid clean TSV line {}: expected 9 columns", line_no));
    }
    let turns = parts[5].parse::<usize>().map_err(|err| err.to_string())?;
    Ok(CleanInputRow {
        index: parts[0].to_string(),
        source: parts[1].to_string(),
        source_id: parts[2].to_string(),
        license: parts[3].to_string(),
        language: parts[4].to_string(),
        turns,
        thread_key: parts[6].to_string(),
        metadata_json: if parts[7].trim().is_empty() {
            "{}".to_string()
        } else {
            parts[7].to_string()
        },
        text: unescape_tsv_text(parts[8]),
    })
}

fn reject_line(index: &str, reason: &'static str) -> String {
    format!("{}\treject\t{}\n", index, reason)
}

fn hard_reject_reason(row: &CleanInputRow, policy: &CleanPolicyRuntime) -> Option<&'static str> {
    let license = row.license.trim().to_ascii_lowercase();
    if !policy.allowed_licenses.contains(&license) {
        return Some("license_not_allowed");
    }
    let language = normalize_language(&row.language);
    if !language.is_empty() && !policy.allowed_languages.contains(&language) {
        return Some("language_not_allowed");
    }
    if row.turns < 2 {
        return Some("too_few_turns");
    }
    if row.turns > 24 {
        return Some("too_many_turns");
    }
    let char_count = row.text.chars().count();
    if char_count < policy.min_chars {
        return Some("too_short");
    }
    if char_count > policy.max_chars {
        return Some("too_long");
    }
    let lower = row.text.to_ascii_lowercase();
    if contains_email(&row.text) || contains_secret(&lower) {
        return Some("pii_or_secret");
    }
    if contains_financial_market_advice(&lower) {
        return Some("financial_market_advice");
    }
    if contains_bad_code_artifact(&lower) {
        return Some("bad_code_artifact");
    }
    if contains_mojibake(&row.text) {
        return Some("mojibake");
    }
    if contains_child_persona_roleplay(&lower) {
        return Some("child_persona_roleplay");
    }
    if contains_unsafe_security(&lower) {
        return Some("unsafe_security");
    }
    if is_url_only(&row.text) {
        return Some("url_only");
    }
    if repetition_ratio(&lower) > 0.35 {
        return Some("repetition");
    }
    if bad_unicode_ratio(&row.text) > 0.01 {
        return Some("bad_unicode");
    }
    None
}

fn score_dialogue(text: &str, lower: &str, turns: usize) -> (f64, &'static str, Vec<&'static str>) {
    let mut score = 0.25f64;
    let mut category = "general";
    let mut tags: Vec<&'static str> = Vec::new();
    if turns >= 4 {
        score += 0.12;
        tags.push("multi_turn");
    }
    if text.chars().count() >= 250 {
        score += 0.10;
        tags.push("substantive");
    }
    if text.contains('?') || text.contains('？') || lower.contains("how") || lower.contains("why") || text.contains("怎么") {
        score += 0.05;
        tags.push("question_answer");
    }
    if has_code_hint(text, lower) {
        score += 0.20;
        category = if has_debug_failure_hint(lower) { "debug" } else { "engineering" };
        tags.push("code_or_debug");
    }
    if has_any(text, &["报错", "错误", "异常", "堆栈", "调试", "复现", "修复", "回归测试", "单元测试", "编译失败", "运行失败"]) {
        score += 0.22;
        category = "debug";
        tags.push("zh_debug");
    }
    if has_any(lower, &["tool", "function call", "schema", "json schema", "argument", "sandbox", "terminal", "command"]) {
        score += 0.12;
        category = "tool_calling";
        tags.push("tool_use");
    }
    if has_any(text, &["工具调用", "工具", "命令", "终端", "接口", "函数调用", "JSON", "模式", "沙箱"]) {
        score += 0.12;
        category = "tool_calling";
        tags.push("zh_tool_use");
    }
    if has_any(lower, &["cve", "cwe", "vulnerability", "sanitize", "injection", "xss", "sqli", "ssrf", "auth bypass", "access control", "exploitability", "security audit", "patch diff", "secure coding"]) {
        score += 0.16;
        category = "security_defensive";
        tags.push("security");
    }
    if has_any(text, &["漏洞", "注入", "越权", "鉴权", "权限绕过", "安全审计", "攻击面", "补丁", "修补", "XSS", "SSRF", "SQL注入"]) {
        score += 0.16;
        category = "security_defensive";
        tags.push("zh_security");
    }
    if has_any(lower, &["prove", "derive", "step by step", "why", "because", "therefore"]) {
        score += 0.08;
        tags.push("reasoning");
    }
    if has_any(text, &["为什么", "原因", "步骤", "推导", "证明", "因为", "所以", "怎么"]) {
        score += 0.08;
        tags.push("zh_reasoning");
    }
    if has_correction_pattern(lower) {
        score += 0.10;
        tags.push("repair");
    }
    if generic_reply_ratio(text) > 0.45 {
        score -= 0.20;
    }
    (score.clamp(0.0, 1.0), category, tags)
}

fn has_code_hint(text: &str, lower: &str) -> bool {
    text.contains("```")
        || has_any(
            lower,
            &[
                "traceback",
                "pytest",
                "unittest",
                "def ",
                "class ",
                "function",
                "typeerror",
                "valueerror",
                "compiler",
                "runtime error",
                "stack trace",
                "exception",
            ],
        )
}

fn has_debug_failure_hint(lower: &str) -> bool {
    has_any(lower, &["fail", "error", "traceback", "pytest", "exception"])
}

fn has_correction_pattern(lower: &str) -> bool {
    has_any(
        lower,
        &[
            "actually",
            "correction",
            "fixed bug",
            "fixing bug",
            "fix bug",
            "fixed issue",
            "fixing issue",
            "fix issue",
            "fixed error",
            "fixing error",
            "fix error",
            "regression test",
            "root cause",
            "i was wrong",
        ],
    )
}

fn generic_reply_ratio(text: &str) -> f64 {
    let mut assistant = 0usize;
    let mut generic = 0usize;
    for segment in text.split("\n\n") {
        let trimmed = segment.trim();
        if let Some(content) = trimmed.strip_prefix("Assistant:") {
            assistant += 1;
            let lower = content.trim().to_ascii_lowercase();
            if ["sure", "yes", "ok", "i can help with that"].contains(&lower.as_str())
                || lower.chars().count() < 25
            {
                generic += 1;
            }
        }
    }
    if assistant == 0 {
        0.0
    } else {
        generic as f64 / assistant as f64
    }
}

fn language_group_for_text(language: &str, text: &str) -> String {
    let language = normalize_language(language);
    if language == "zh" {
        return "zh".to_string();
    }
    if language == "en" {
        return "en".to_string();
    }
    let mut cjk = 0usize;
    let mut ascii_letters = 0usize;
    for ch in text.chars() {
        if ('\u{4e00}'..='\u{9fff}').contains(&ch) {
            cjk += 1;
        }
        if ch.is_ascii_alphabetic() {
            ascii_letters += 1;
        }
    }
    if cjk >= 8 && (cjk as f64) >= (ascii_letters as f64 * 0.2) {
        "zh".to_string()
    } else {
        "en".to_string()
    }
}

fn build_training_json_line(
    row: &CleanInputRow,
    language: &str,
    category: &str,
    score: f64,
    tags: &[&str],
) -> String {
    let metadata_json = if row.metadata_json.trim().is_empty() {
        "{}"
    } else {
        row.metadata_json.trim()
    };
    format!(
        "{{\"text\":\"{}\",\"language\":\"{}\",\"source\":\"{}\",\"source_id\":\"{}\",\"license\":\"{}\",\"category\":\"{}\",\"quality_score\":{},\"quality_tags\":[{}],\"metadata\":{}}}\n",
        json_escape(&row.text),
        json_escape(language),
        json_escape(&row.source),
        json_escape(&row.source_id),
        json_escape(&row.license),
        json_escape(category),
        trim_float(score),
        tags.iter()
            .map(|tag| format!("\"{}\"", json_escape(tag)))
            .collect::<Vec<_>>()
            .join(","),
        metadata_json,
    )
}

fn fingerprint_key(text: &str) -> (u64, usize) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let normalized = text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    let mut hasher = DefaultHasher::new();
    normalized.hash(&mut hasher);
    (hasher.finish(), normalized.len())
}

fn unescape_tsv_text(value: &str) -> String {
    value.replace("\\n", "\n")
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => escaped.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => escaped.push(ch),
        }
    }
    escaped
}

fn trim_float(value: f64) -> String {
    let mut raw = format!("{:.4}", value);
    while raw.contains('.') && raw.ends_with('0') {
        raw.pop();
    }
    if raw.ends_with('.') {
        raw.push('0');
    }
    raw
}

fn has_any(text: &str, patterns: &[&str]) -> bool {
    patterns.iter().any(|pattern| text.contains(pattern))
}

fn contains_child_persona_roleplay(lower: &str) -> bool {
    (has_any(lower, &["pretend", "roleplay", "act"])
        && has_any(lower, &["6 year old", "six year old", "7 year old", "seven year old", "8 year old", "eight year old"])
        && has_any(lower, &["girl", "boy", "child"]))
        || lower.contains("call me mommy")
        || lower.contains("baby girl")
}

fn filter_tsv(
    input: &Path,
    output: &Path,
    min_chars: usize,
    max_chars: usize,
    allowed_licenses: &HashSet<String>,
    allowed_languages: &HashSet<String>,
) -> Result<(), String> {
    let input_file = File::open(input).map_err(|err| err.to_string())?;
    let output_file = File::create(output).map_err(|err| err.to_string())?;
    filter_reader(
        BufReader::new(input_file),
        BufWriter::new(output_file),
        min_chars,
        max_chars,
        allowed_licenses,
        allowed_languages,
    )
}

fn filter_stdin(
    min_chars: usize,
    max_chars: usize,
    allowed_licenses: &HashSet<String>,
    allowed_languages: &HashSet<String>,
) -> Result<(), String> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    filter_reader(
        stdin.lock(),
        BufWriter::new(stdout.lock()),
        min_chars,
        max_chars,
        allowed_licenses,
        allowed_languages,
    )
}

fn filter_batch_stdin(
    min_chars: usize,
    max_chars: usize,
    allowed_licenses: &HashSet<String>,
    allowed_languages: &HashSet<String>,
) -> Result<(), String> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    filter_batch_stream(
        stdin.lock(),
        BufWriter::new(stdout.lock()),
        min_chars,
        max_chars,
        allowed_licenses,
        allowed_languages,
    )
}

fn clean_batch_stdin(policy: CleanPolicyRuntime) -> Result<(), String> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut state = CleanState::default();
    clean_batch_stream(
        stdin.lock(),
        BufWriter::new(stdout.lock()),
        &policy,
        &mut state,
    )
}

fn csv_set(raw: &str) -> HashSet<String> {
    raw.split(',')
        .map(|item| item.trim().to_ascii_lowercase())
        .filter(|item| !item.is_empty())
        .collect()
}

fn reject(reason: &'static str) -> Verdict {
    Verdict {
        accepted: false,
        reason,
    }
}

fn normalize_language(raw: &str) -> String {
    let value = raw.trim().to_ascii_lowercase();
    match value.as_str() {
        "zh" | "zh-cn" | "zh-hans" | "zh-hant" | "zho" | "cmn" | "chinese"
        | "simplified chinese" | "traditional chinese" | "中文" | "cn" => "zh".to_string(),
        "en" | "eng" | "english" => "en".to_string(),
        _ => value,
    }
}

fn contains_email(text: &str) -> bool {
    for token in text.split_whitespace() {
        let trimmed = token.trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '@' && ch != '.' && ch != '_' && ch != '-' && ch != '+');
        if let Some(at) = trimmed.find('@') {
            if at > 0 && trimmed[at + 1..].contains('.') {
                return true;
            }
        }
    }
    false
}

fn contains_secret(lower: &str) -> bool {
    if lower.contains("akia") {
        return true;
    }
    if lower.contains("sk-") {
        return true;
    }
    for key in ["api_key", "api-key", "secret", "token", "password"] {
        if let Some(pos) = lower.find(key) {
            let tail = &lower[pos + key.len()..];
            if tail.trim_start().starts_with(':') || tail.trim_start().starts_with('=') {
                return true;
            }
        }
    }
    false
}

fn contains_bad_code_artifact(lower: &str) -> bool {
    lower.contains("flask(name)")
        || lower.contains("if name == 'main'")
        || lower.contains("if name == \"main\"")
        || lower.contains("python copy code")
        || lower.contains("blenderpy")
}

fn contains_mojibake(text: &str) -> bool {
    let patterns = [
        "�", "Ã", "â€", "鈥", "檚", "搒", "揷", "榚", "茅", "铆", "€", "缁", "涓€", "璇",
        "鐢", "鍚", "鏁", "鐓", "楠", "姹",
    ];
    patterns.iter().any(|pattern| text.contains(pattern))
}

fn is_url_only(text: &str) -> bool {
    let mut count = 0usize;
    for item in text.split_whitespace() {
        if !(item.starts_with("http://") || item.starts_with("https://")) {
            return false;
        }
        count += 1;
    }
    count > 0 && count <= 3
}

fn bad_unicode_ratio(text: &str) -> f64 {
    let mut total = 0usize;
    let mut bad = 0usize;
    for ch in text.chars() {
        total += 1;
        if ch.is_control() && ch != '\n' && ch != '\t' && ch != '\r' {
            bad += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        bad as f64 / total as f64
    }
}

fn contains_unsafe_security(lower: &str) -> bool {
    let sensitive = ["passcode", "phone", "pin", "password"];
    let brute = [
        "guess",
        "brute force",
        "generate combinations",
        "test each combination",
        "try every",
    ];
    sensitive.iter().any(|left| lower.contains(left)) && brute.iter().any(|right| lower.contains(right))
}

fn contains_financial_market_advice(lower: &str) -> bool {
    [
        "cryptocurrency",
        "crypto",
        "bitcoin",
        "ethereum",
        "stock alert",
        "stock price",
        "nvidia stock",
        "nvda",
        "investing plan",
        "investment decision",
        "financial advisor",
        "portfolio",
        "market timing",
        "market volatility",
        "dollar-cost averaging",
    ]
    .iter()
    .any(|pattern| lower.contains(pattern))
}

fn repetition_ratio(lower: &str) -> f64 {
    let mut token_count = 0usize;
    let mut counts = std::collections::HashMap::<&str, usize>::new();
    for token in lower
        .split(|ch: char| !ch.is_alphanumeric() && ch != '_')
        .filter(|item| !item.is_empty())
    {
        token_count += 1;
        *counts.entry(token).or_insert(0) += 1;
    }
    if token_count < 20 {
        return 0.0;
    }
    let repeated: usize = counts
        .iter()
        .filter(|(token, count)| token.len() > 2 && **count > 4)
        .map(|(_, count)| *count)
        .sum();
    repeated as f64 / token_count as f64
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let result = match args.get(1).map(|item| item.as_str()) {
        Some("health") => {
            println!("ok");
            Ok(())
        }
        Some("filter-tsv") => {
            if args.len() != 8 {
                Err("usage: dopa_dialogue_filter filter-tsv <input> <output> <min_chars> <max_chars> <allowed_licenses_csv> <allowed_langs_csv>".to_string())
            } else {
                let min_chars = args[4].parse::<usize>().map_err(|err| err.to_string());
                let max_chars = args[5].parse::<usize>().map_err(|err| err.to_string());
                match (min_chars, max_chars) {
                    (Ok(min_chars), Ok(max_chars)) => filter_tsv(
                        Path::new(&args[2]),
                        Path::new(&args[3]),
                        min_chars,
                        max_chars,
                        &csv_set(&args[6]),
                        &csv_set(&args[7]),
                    ),
                    (Err(err), _) => Err(err),
                    (_, Err(err)) => Err(err),
                }
            }
        }
        Some("filter-stdin") => {
            if args.len() != 6 {
                Err("usage: dopa_dialogue_filter filter-stdin <min_chars> <max_chars> <allowed_licenses_csv> <allowed_langs_csv>".to_string())
            } else {
                let min_chars = args[2].parse::<usize>().map_err(|err| err.to_string());
                let max_chars = args[3].parse::<usize>().map_err(|err| err.to_string());
                match (min_chars, max_chars) {
                    (Ok(min_chars), Ok(max_chars)) => filter_stdin(
                        min_chars,
                        max_chars,
                        &csv_set(&args[4]),
                        &csv_set(&args[5]),
                    ),
                    (Err(err), _) => Err(err),
                    (_, Err(err)) => Err(err),
                }
            }
        }
        Some("filter-batch-stdin") => {
            if args.len() != 6 {
                Err("usage: dopa_dialogue_filter filter-batch-stdin <min_chars> <max_chars> <allowed_licenses_csv> <allowed_langs_csv>".to_string())
            } else {
                let min_chars = args[2].parse::<usize>().map_err(|err| err.to_string());
                let max_chars = args[3].parse::<usize>().map_err(|err| err.to_string());
                match (min_chars, max_chars) {
                    (Ok(min_chars), Ok(max_chars)) => filter_batch_stdin(
                        min_chars,
                        max_chars,
                        &csv_set(&args[4]),
                        &csv_set(&args[5]),
                    ),
                    (Err(err), _) => Err(err),
                    (_, Err(err)) => Err(err),
                }
            }
        }
        Some("clean-batch-stdin") => {
            if args.len() != 8 {
                Err("usage: dopa_dialogue_filter clean-batch-stdin <min_chars> <max_chars> <min_score> <allowed_licenses_csv> <allowed_langs_csv> <target_categories_csv>".to_string())
            } else {
                let min_chars = args[2].parse::<usize>().map_err(|err| err.to_string());
                let max_chars = args[3].parse::<usize>().map_err(|err| err.to_string());
                let min_score = args[4].parse::<f64>().map_err(|err| err.to_string());
                match (min_chars, max_chars, min_score) {
                    (Ok(min_chars), Ok(max_chars), Ok(min_score)) => clean_batch_stdin(CleanPolicyRuntime::new(
                        min_chars,
                        max_chars,
                        min_score,
                        csv_set(&args[5]),
                        csv_set(&args[6]),
                        csv_set(&args[7]),
                    )),
                    (Err(err), _, _) => Err(err),
                    (_, Err(err), _) => Err(err),
                    (_, _, Err(err)) => Err(err),
                }
            }
        }
        _ => Err("usage: dopa_dialogue_filter <health|filter-tsv|filter-stdin|filter-batch-stdin|clean-batch-stdin>".to_string()),
    };
    if let Err(err) = result {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn allowed_licenses() -> HashSet<String> {
        csv_set("apache-2.0,odc-by,cc-by-4.0,mit")
    }

    fn allowed_languages() -> HashSet<String> {
        csv_set("en,eng,english,zh,zh-cn,zh-hans,zho,cmn,chinese,simplified chinese")
    }

    #[test]
    fn rejects_email_and_secret_like_text() {
        let verdict = prefilter_text(
            "apache-2.0",
            "en",
            2,
            "User: email me at owner@example.com\nAssistant: use api_key=redacted-test-key",
            20,
            5000,
            &allowed_licenses(),
            &allowed_languages(),
        );

        assert_eq!(verdict.reason, "pii_or_secret");
        assert!(!verdict.accepted);
    }

    #[test]
    fn accepts_chinese_language_codes_when_text_is_substantive() {
        let verdict = prefilter_text(
            "apache-2.0",
            "zho",
            2,
            "User: 怎么根据 pytest 报错定位 Python 函数里的问题?\nAssistant: 先读堆栈最后一行, 再缩小输入, 最后补一个回归测试。",
            20,
            5000,
            &allowed_licenses(),
            &allowed_languages(),
        );

        assert!(verdict.accepted);
        assert_eq!(verdict.reason, "accepted");
    }

    #[test]
    fn filter_reader_writes_one_verdict_per_input_line() {
        let input = b"0\tapache-2.0\ten\t2\tUser: How do I debug pytest? Assistant: Read the stack trace.\n1\tbad-license\ten\t2\tUser: good text Assistant: good answer\n";
        let mut output = Vec::new();

        filter_reader(
            BufReader::new(&input[..]),
            &mut output,
            20,
            5000,
            &allowed_licenses(),
            &allowed_languages(),
        )
        .unwrap();
        let body = String::from_utf8(output).unwrap();

        assert!(body.contains("0\taccept\taccepted"));
        assert!(body.contains("1\treject\tlicense_not_allowed"));
    }

    #[test]
    fn filter_batch_stream_writes_and_flushes_one_batch() {
        let input = b"2\n0\tapache-2.0\ten\t2\tUser: How do I debug pytest? Assistant: Read the stack trace.\n1\tbad-license\ten\t2\tUser: good text Assistant: good answer\n";
        let mut output = Vec::new();

        filter_batch_stream(
            BufReader::new(&input[..]),
            &mut output,
            20,
            5000,
            &allowed_licenses(),
            &allowed_languages(),
        )
        .unwrap();
        let body = String::from_utf8(output).unwrap();

        assert!(body.contains("0\taccept\taccepted"));
        assert!(body.contains("1\treject\tlicense_not_allowed"));
    }

    #[test]
    fn clean_batch_stream_outputs_accepted_jsonl_and_rejects_duplicate_thread() {
        let input = b"2\n0\tunit/source\trust-clean\tapache-2.0\ten\t2\tunit/source:tree-1\t{\"language\":\"en\",\"message_tree_id\":\"tree-1\"}\tUser: How do I debug pytest fixture failures?\\n\\nAssistant: Read the traceback, isolate fixture state, reproduce the smallest failing case, then add a regression test.\n1\tunit/source\trust-clean-dup\tapache-2.0\ten\t2\tunit/source:tree-1\t{\"language\":\"en\",\"message_tree_id\":\"tree-1\"}\tUser: How do I debug pytest fixture failures?\\n\\nAssistant: Read the traceback, isolate fixture state, reproduce the smallest failing case, then add a regression test.\n";
        let mut output = Vec::new();
        let policy = CleanPolicyRuntime::new(
            80,
            12_000,
            0.6,
            csv_set("apache-2.0,odc-by,cc-by-4.0,mit"),
            csv_set("en,eng,english,zh,zh-cn,zh-hans,zho,cmn,chinese,simplified chinese"),
            csv_set("debug,engineering,tool_calling,security_defensive"),
        );
        let mut state = CleanState::default();

        clean_batch_stream(BufReader::new(&input[..]), &mut output, &policy, &mut state).unwrap();
        let body = String::from_utf8(output).unwrap();

        assert!(body.contains("0\taccept\taccepted\ten\tdebug"));
        assert!(body.contains("\"category\":\"debug\""));
        assert!(body.contains("\"quality_score\":0.6"));
        assert!(body.contains("1\treject\tduplicate_thread"));
    }

    #[test]
    fn filter_tsv_writes_one_verdict_per_input_line() {
        let temp = env::temp_dir().join(format!("dopa-dialogue-filter-{}", std::process::id()));
        fs::create_dir_all(&temp).unwrap();
        let input = temp.join("input.tsv");
        let output = temp.join("output.tsv");
        fs::write(
            &input,
            "0\tapache-2.0\ten\t2\tUser: How do I debug pytest? Assistant: Read the stack trace.\n1\tbad-license\ten\t2\tUser: good text Assistant: good answer\n",
        )
        .unwrap();

        filter_tsv(
            &input,
            &output,
            20,
            5000,
            &allowed_licenses(),
            &allowed_languages(),
        )
        .unwrap();
        let body = fs::read_to_string(&output).unwrap();

        assert!(body.contains("0\taccept\taccepted"));
        assert!(body.contains("1\treject\tlicense_not_allowed"));
    }
}

