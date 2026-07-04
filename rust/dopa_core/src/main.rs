use std::cmp::Ordering;
use std::env;
use std::fs;
use std::path::Path;

const DIM: usize = 128;

fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            cur.push(ch.to_ascii_lowercase());
        } else if !cur.is_empty() {
            if cur.len() > 3 && cur.ends_with('s') {
                out.push(cur[..cur.len() - 1].to_string());
            }
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        if cur.len() > 3 && cur.ends_with('s') {
            out.push(cur[..cur.len() - 1].to_string());
        }
        out.push(cur);
    }
    out
}

fn fnv1a(token: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in token.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn embed(text: &str) -> [f32; DIM] {
    let mut vec = [0.0f32; DIM];
    for token in tokenize(text) {
        let hash = fnv1a(&token);
        let idx = (hash as usize) % DIM;
        let sign = if (hash >> 63) == 0 { 1.0 } else { -1.0 };
        vec[idx] += sign;
    }
    let mut norm = 0.0f32;
    for value in vec {
        norm += value * value;
    }
    norm = norm.sqrt();
    if norm > 0.0 {
        for value in vec.iter_mut() {
            *value /= norm;
        }
    }
    vec
}

fn lexical_overlap(query: &str, text: &str) -> f32 {
    let query_tokens = tokenize(query);
    if query_tokens.is_empty() {
        return 0.0;
    }
    let text_tokens = tokenize(text);
    let mut hits = 0usize;
    for token in query_tokens.iter() {
        if text_tokens.iter().any(|candidate| candidate == token) {
            hits += 1;
        }
    }
    hits as f32 / query_tokens.len() as f32
}

fn score(query: &str, text: &str) -> f32 {
    let q = embed(query);
    let t = embed(text);
    let mut dense = 0.0f32;
    for idx in 0..DIM {
        dense += q[idx] * t[idx];
    }
    0.70 * dense + 0.30 * lexical_overlap(query, text)
}

fn byte_encode(text: &str, add_bos: bool, add_eos: bool) {
    let mut first = true;
    if add_bos {
        print!("1");
        first = false;
    }
    for byte in text.as_bytes() {
        if !first {
            print!(" ");
        }
        print!("{}", u16::from(*byte) + 4);
        first = false;
    }
    if add_eos {
        if !first {
            print!(" ");
        }
        print!("2");
    }
    println!();
}

fn byte_ids(text: &str, add_bos: bool, add_eos: bool) -> Vec<u16> {
    let mut ids = Vec::with_capacity(text.len() + usize::from(add_bos) + usize::from(add_eos));
    if add_bos {
        ids.push(1);
    }
    for byte in text.as_bytes() {
        ids.push(u16::from(*byte) + 4);
    }
    if add_eos {
        ids.push(2);
    }
    ids
}

fn print_json_array(ids: &[u16]) {
    print!("[");
    for (idx, id) in ids.iter().enumerate() {
        if idx > 0 {
            print!(",");
        }
        print!("{id}");
    }
    println!("]");
}

fn byte_encode_lines(path: &Path, add_bos_first: bool, add_eos_each: bool) -> Result<(), String> {
    let body = fs::read_to_string(path).map_err(|err| err.to_string())?;
    for (idx, line) in body.lines().enumerate() {
        let ids = byte_ids(line, add_bos_first && idx == 0, add_eos_each);
        print_json_array(&ids);
    }
    Ok(())
}

fn byte_decode(ids: &str) -> Result<(), String> {
    let mut bytes = Vec::new();
    for item in ids.split_whitespace() {
        let id = item.parse::<i64>().map_err(|err| err.to_string())?;
        if id >= 4 {
            if id > 259 {
                return Err(format!("byte token id out of range: {id}"));
            }
            bytes.push((id - 4) as u8);
        }
    }
    let text = String::from_utf8_lossy(&bytes);
    print!("{text}");
    Ok(())
}

fn encode_skeleton_text(text: &str, vocab_size: usize, max_len: usize) -> Result<Vec<usize>, String> {
    if vocab_size <= 4 {
        return Err("vocab_size must be greater than 4".to_string());
    }
    let mut ids = Vec::with_capacity(max_len.max(2));
    ids.push(1);
    let body_len = max_len.saturating_sub(2);
    for byte in text.as_bytes().iter().take(body_len) {
        ids.push((*byte as usize % (vocab_size - 4)) + 4);
    }
    ids.push(2);
    while ids.len() < max_len {
        ids.push(0);
    }
    ids.truncate(max_len);
    Ok(ids)
}

fn sorted_json_text(input: &str) -> String {
    let trimmed = input.trim();
    if trimmed.starts_with('{') {
        let inner = &trimmed[1..trimmed.len().saturating_sub(1)];
        let mut pairs: Vec<(String, String)> = Vec::new();
        for part in inner.split(',') {
            let mut kv = part.splitn(2, ':');
            if let (Some(k), Some(v)) = (kv.next(), kv.next()) {
                pairs.push((k.trim().to_string(), v.trim().to_string()));
            }
        }
        if pairs.len() > 1 && pairs.iter().all(|(key, value)| key.starts_with('"') && value.chars().all(|ch| !ch.is_whitespace())) {
            pairs.sort_by(|a, b| a.0.cmp(&b.0));
            let mut out = String::from("{");
            for (idx, (key, value)) in pairs.iter().enumerate() {
                if idx > 0 {
                    out.push_str(", ");
                }
                out.push_str(key);
                out.push_str(": ");
                out.push_str(value);
            }
            out.push('}');
            return out;
        }
    }
    input.to_string()
}

fn skeleton_encode_json(path: &Path, vocab_size: usize, max_len: usize) -> Result<(), String> {
    let body = fs::read_to_string(path).map_err(|err| err.to_string())?;
    let text = sorted_json_text(&body);
    let ids = encode_skeleton_text(&text, vocab_size, max_len)?;
    for (idx, id) in ids.iter().enumerate() {
        if idx > 0 {
            print!(" ");
        }
        print!("{id}");
    }
    println!();
    Ok(())
}

fn rank_text(query: &str, input: &Path, limit: usize) -> Result<(), String> {
    let body = fs::read_to_string(input).map_err(|err| err.to_string())?;
    let mut rows: Vec<(String, String, f32)> = Vec::new();
    for line in body.lines() {
        let mut parts = line.splitn(2, '\t');
        let id = parts.next().unwrap_or("").to_string();
        let text = parts.next().unwrap_or("").to_string();
        if !id.is_empty() {
            let item_score = score(query, &text);
            rows.push((id, text, item_score));
        }
    }
    rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
    for (id, text, item_score) in rows.into_iter().take(limit) {
        println!("{}\t{:.6}\t{}", id, item_score, text.replace('\t', " ").replace('\n', " "));
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let result = match args.get(1).map(|item| item.as_str()) {
        Some("health") => {
            println!("ok");
            Ok(())
        }
        Some("rank-text") => {
            if args.len() != 5 {
                Err("usage: dopa_core rank-text <query> <input_tsv> <limit>".to_string())
            } else {
                let limit = args[4].parse::<usize>().map_err(|err| err.to_string());
                match limit {
                    Ok(limit) => rank_text(&args[2], Path::new(&args[3]), limit),
                    Err(err) => Err(err),
                }
            }
        }
        Some("byte-encode") => {
            if args.len() != 5 {
                Err("usage: dopa_core byte-encode <text> <add_bos:0|1> <add_eos:0|1>".to_string())
            } else {
                byte_encode(&args[2], args[3] == "1", args[4] == "1");
                Ok(())
            }
        }
        Some("byte-encode-lines") => {
            if args.len() != 5 {
                Err("usage: dopa_core byte-encode-lines <text_path> <add_bos_first:0|1> <add_eos_each:0|1>".to_string())
            } else {
                byte_encode_lines(Path::new(&args[2]), args[3] == "1", args[4] == "1")
            }
        }
        Some("byte-decode") => {
            if args.len() != 3 {
                Err("usage: dopa_core byte-decode <space_separated_ids>".to_string())
            } else {
                byte_decode(&args[2])
            }
        }
        Some("skeleton-encode-json") => {
            if args.len() != 5 {
                Err("usage: dopa_core skeleton-encode-json <json_path> <vocab_size> <max_len>".to_string())
            } else {
                match (args[3].parse::<usize>(), args[4].parse::<usize>()) {
                    (Ok(vocab_size), Ok(max_len)) => skeleton_encode_json(Path::new(&args[2]), vocab_size, max_len),
                    (Err(err), _) => Err(err.to_string()),
                    (_, Err(err)) => Err(err.to_string()),
                }
            }
        }
        _ => Err("usage: dopa_core <health|rank-text|byte-encode|byte-encode-lines|byte-decode|skeleton-encode-json>".to_string()),
    };
    if let Err(err) = result {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
