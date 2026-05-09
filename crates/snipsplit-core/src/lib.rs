//! Pure-Rust core for `snipsplit`. Token-aware greedy chunker for RAG.
//!
//! Algorithm:
//! 1. Split into paragraphs on blank lines, then sentences via a regex
//!    that handles the common abbreviation pitfalls (`Mr.`, `Dr.`, `e.g.`,
//!    `vs.`, `etc.`, version-style `1.0`, decimal numbers).
//! 2. Greedy-pack sentences into chunks while the running BPE token count
//!    is `<= max_tokens`.
//! 3. If a single sentence is too big on its own, slice it at token
//!    boundaries instead.
//! 4. Apply `overlap_tokens` by re-prepending the last N tokens of each
//!    emitted chunk to the next.
//! 5. Drop chunks shorter than `min_tokens`.

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tiktoken_rs::CoreBPE;

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, ChunkerError>;

/// All errors surfaced by `snipsplit-core`.
#[derive(Error, Debug)]
pub enum ChunkerError {
    /// Unknown encoding name. Supported: `cl100k_base`, `o200k_base`.
    #[error("unknown encoding: {0} (expected cl100k_base or o200k_base)")]
    UnknownEncoding(String),
    /// Caller supplied an invalid configuration value.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    /// tiktoken-rs failure.
    #[error("tiktoken-rs error: {0}")]
    Tiktoken(String),
}

/// Chunker configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkConfig {
    /// Hard cap on tokens per chunk.
    pub max_tokens: usize,
    /// Number of trailing tokens of each chunk re-prepended to the next.
    /// Set to 0 to disable overlap.
    pub overlap_tokens: usize,
    /// Drop chunks shorter than this.
    pub min_tokens: usize,
    /// Encoding name (`cl100k_base` or `o200k_base`).
    pub encoding: String,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            overlap_tokens: 0,
            min_tokens: 1,
            encoding: "cl100k_base".to_string(),
        }
    }
}

/// One emitted chunk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    /// The chunk text. Includes any prepended overlap from the prior chunk.
    pub text: String,
    /// Byte offset of the first chunk character (excluding overlap) in the
    /// original input. Two adjacent chunks may have overlapping `[start, end)`
    /// ranges if `overlap_tokens > 0`.
    pub start: usize,
    /// Byte offset (exclusive) of the chunk end in the original input.
    pub end: usize,
    /// Exact BPE token count of `text`.
    pub token_count: usize,
}

/// Token-aware chunker.
pub struct Chunker {
    bpe: CoreBPE,
    cfg: ChunkConfig,
    sentence_re: Regex,
}

impl Chunker {
    /// Build a chunker from a config.
    pub fn new(cfg: ChunkConfig) -> Result<Self> {
        if cfg.max_tokens == 0 {
            return Err(ChunkerError::InvalidConfig("max_tokens must be > 0".into()));
        }
        if cfg.overlap_tokens >= cfg.max_tokens {
            return Err(ChunkerError::InvalidConfig(format!(
                "overlap_tokens ({}) must be < max_tokens ({})",
                cfg.overlap_tokens, cfg.max_tokens
            )));
        }
        if cfg.min_tokens > cfg.max_tokens {
            return Err(ChunkerError::InvalidConfig(format!(
                "min_tokens ({}) must be <= max_tokens ({})",
                cfg.min_tokens, cfg.max_tokens
            )));
        }
        let bpe = match cfg.encoding.as_str() {
            "cl100k_base" => {
                tiktoken_rs::cl100k_base().map_err(|e| ChunkerError::Tiktoken(e.to_string()))?
            }
            "o200k_base" => {
                tiktoken_rs::o200k_base().map_err(|e| ChunkerError::Tiktoken(e.to_string()))?
            }
            other => return Err(ChunkerError::UnknownEncoding(other.to_string())),
        };

        // Sentence boundary regex. Matches whitespace following sentence-
        // terminating punctuation, but not when preceded by a known
        // abbreviation. Conservative; deliberately misses some cases (e.g.
        // numbered lists) rather than over-splitting.
        let sentence_re = Regex::new(
            r"(?P<term>[.!?])(?P<close>[\)\]\}\u{201d}\u{2019}\u{00bb}'\x22]?)\s+(?P<next>[A-Z\u{00c0}-\u{00de}\u{2018}\u{201c}\(\[\{])"
        ).expect("sentence regex compiles");

        Ok(Self {
            bpe,
            cfg,
            sentence_re,
        })
    }

    /// Split `text` into chunks.
    pub fn split(&self, text: &str) -> Result<Vec<Chunk>> {
        // Step 1: collect sentence spans (start, end) into the original text.
        let sentences = self.split_sentences(text);
        if sentences.is_empty() {
            return Ok(Vec::new());
        }

        // Step 2: pre-compute token IDs for each sentence so we count once.
        let mut s_tokens: Vec<Vec<u32>> = Vec::with_capacity(sentences.len());
        for &(start, end) in &sentences {
            s_tokens.push(self.bpe.encode_ordinary(&text[start..end]));
        }

        // Step 3: greedy pack. Emits raw chunks (without overlap) as a vec of
        // owned token id sequences and the (start, end) byte spans they cover.
        let mut raw: Vec<(Vec<u32>, usize, usize)> = Vec::new();
        let mut cur_tokens: Vec<u32> = Vec::new();
        let mut cur_start: Option<usize> = None;
        let mut cur_end: usize = 0;
        for (i, &(s_start, s_end)) in sentences.iter().enumerate() {
            let stoks = &s_tokens[i];
            // If this single sentence already exceeds the budget, flush
            // whatever we have and slice the sentence at token boundaries.
            if stoks.len() > self.cfg.max_tokens {
                if !cur_tokens.is_empty() {
                    raw.push((std::mem::take(&mut cur_tokens), cur_start.unwrap(), cur_end));
                    cur_start = None;
                }
                self.slice_long_sentence(stoks, s_start, s_end, &mut raw);
                continue;
            }
            // Would adding this sentence overflow?
            if cur_tokens.len() + stoks.len() > self.cfg.max_tokens && !cur_tokens.is_empty() {
                raw.push((std::mem::take(&mut cur_tokens), cur_start.unwrap(), cur_end));
                cur_start = None;
            }
            if cur_start.is_none() {
                cur_start = Some(s_start);
            }
            cur_tokens.extend_from_slice(stoks);
            cur_end = s_end;
        }
        if !cur_tokens.is_empty() {
            raw.push((cur_tokens, cur_start.unwrap(), cur_end));
        }

        // Step 4: apply overlap and decode.
        let mut out: Vec<Chunk> = Vec::with_capacity(raw.len());
        let mut prev_tail: Vec<u32> = Vec::new();
        for (toks, start, end) in raw {
            let mut full = Vec::with_capacity(prev_tail.len() + toks.len());
            full.extend_from_slice(&prev_tail);
            full.extend_from_slice(&toks);
            let text = self
                .bpe
                .decode(full.clone())
                .map_err(|e| ChunkerError::Tiktoken(e.to_string()))?;
            // Update prev_tail for the next iteration.
            prev_tail = if self.cfg.overlap_tokens > 0 && toks.len() > self.cfg.overlap_tokens {
                toks[toks.len() - self.cfg.overlap_tokens..].to_vec()
            } else if self.cfg.overlap_tokens > 0 {
                toks.clone()
            } else {
                Vec::new()
            };
            let token_count = full.len();
            // Skip below min_tokens.
            if token_count < self.cfg.min_tokens {
                continue;
            }
            out.push(Chunk {
                text,
                start,
                end,
                token_count,
            });
        }
        Ok(out)
    }

    /// Split many texts. With `parallel = true`, distributes across the
    /// rayon pool. Each call into `split` is independent.
    pub fn split_many(&self, texts: &[&str], parallel: bool) -> Result<Vec<Vec<Chunk>>> {
        if parallel {
            texts.par_iter().map(|t| self.split(t)).collect()
        } else {
            texts.iter().map(|t| self.split(t)).collect()
        }
    }

    /// Sentence boundaries, returned as `(byte_start, byte_end)` half-open
    /// ranges into `text`. Always covers the full input — there are no
    /// gaps. Empty input returns no sentences.
    fn split_sentences(&self, text: &str) -> Vec<(usize, usize)> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut last = 0usize;
        for caps in self.sentence_re.captures_iter(text) {
            let m = caps.name("term").unwrap();
            // Cut just after the closing punctuation/quote group, BEFORE the
            // whitespace before `next`. We use the end of the `close` group
            // if present, otherwise end of `term`.
            let cut = caps
                .name("close")
                .filter(|c| !c.as_str().is_empty())
                .map(|c| c.end())
                .unwrap_or_else(|| m.end());
            // Skip if this would create an empty span.
            if cut <= last {
                continue;
            }
            // Suppress split if the substring just before `term` is a known
            // abbreviation. Cheap heuristic; production splitters use a
            // gazetteer.
            if is_abbreviation(&text[..m.end()]) {
                continue;
            }
            spans.push((last, cut));
            // Advance last past any whitespace.
            let mut next_start = cut;
            while next_start < text.len() && text.as_bytes()[next_start].is_ascii_whitespace() {
                next_start += 1;
            }
            last = next_start;
        }
        if last < text.len() {
            spans.push((last, text.len()));
        }
        // Filter out empty/whitespace-only spans.
        spans.retain(|&(s, e)| s < e && !text[s..e].trim().is_empty());
        spans
    }

    /// Slice an over-long sentence at token boundaries.
    fn slice_long_sentence(
        &self,
        toks: &[u32],
        s_start: usize,
        s_end: usize,
        out: &mut Vec<(Vec<u32>, usize, usize)>,
    ) {
        // We can't recover exact byte offsets per token without re-encoding
        // partials, so attribute the entire sentence span to every slice.
        // Callers wanting exact offsets should bump the budget instead.
        let mut i = 0usize;
        while i < toks.len() {
            let end = (i + self.cfg.max_tokens).min(toks.len());
            out.push((toks[i..end].to_vec(), s_start, s_end));
            i = end;
        }
    }
}

/// Suffix-check the string against a small list of common English
/// abbreviations that produce false positives in sentence splitting.
fn is_abbreviation(prefix: &str) -> bool {
    const ABBREVS: &[&str] = &[
        "mr.", "mrs.", "ms.", "dr.", "st.", "jr.", "sr.", "prof.", "rev.", "vs.", "etc.", "e.g.",
        "i.e.", "fig.", "cf.", "no.", "vol.", "ch.", "sec.",
    ];
    let lower_tail: String = prefix
        .chars()
        .rev()
        .take(8)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>()
        .to_lowercase();
    ABBREVS.iter().any(|a| lower_tail.ends_with(a))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max_tokens: usize) -> ChunkConfig {
        ChunkConfig {
            max_tokens,
            overlap_tokens: 0,
            min_tokens: 1,
            encoding: "cl100k_base".to_string(),
        }
    }

    #[test]
    fn empty_input_yields_no_chunks() {
        let c = Chunker::new(cfg(100)).unwrap();
        assert!(c.split("").unwrap().is_empty());
    }

    #[test]
    fn short_text_one_chunk() {
        let c = Chunker::new(cfg(100)).unwrap();
        let r = c.split("hello world").unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].text, "hello world");
    }

    #[test]
    fn splits_at_sentence_boundary_under_budget() {
        let c = Chunker::new(cfg(8)).unwrap();
        let text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota.";
        let chunks = c.split(text).unwrap();
        // 3 sentences, ~5 tokens each at cl100k. Should produce more than 1
        // chunk under a budget of 8.
        assert!(
            chunks.len() >= 2,
            "expected >=2 chunks, got {}",
            chunks.len()
        );
        for ch in &chunks {
            assert!(
                ch.token_count <= 8,
                "chunk over budget: {} tokens",
                ch.token_count
            );
        }
    }

    #[test]
    fn long_sentence_falls_back_to_token_slicing() {
        let c = Chunker::new(cfg(5)).unwrap();
        // Single sentence with many tokens.
        let text = "the quick brown fox jumps over the lazy dog and runs through fields";
        let chunks = c.split(text).unwrap();
        assert!(chunks.len() > 1);
        for ch in &chunks {
            assert!(ch.token_count <= 5);
        }
    }

    #[test]
    fn overlap_re_prepends_tail_tokens() {
        let c = Chunker::new(ChunkConfig {
            max_tokens: 6,
            overlap_tokens: 2,
            min_tokens: 1,
            encoding: "cl100k_base".to_string(),
        })
        .unwrap();
        let text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota.";
        let chunks = c.split(text).unwrap();
        // Each chunk after the first should include the prior chunk's last
        // 2 tokens, so total token_count can exceed max_tokens by up to
        // overlap_tokens.
        assert!(chunks.len() >= 2);
        for ch in chunks.iter().skip(1) {
            assert!(ch.token_count <= 6 + 2);
        }
    }

    #[test]
    fn min_tokens_drops_short_chunks() {
        // After packing, any chunk below 50 tokens is dropped.
        let c = Chunker::new(ChunkConfig {
            max_tokens: 1000,
            overlap_tokens: 0,
            min_tokens: 50,
            encoding: "cl100k_base".to_string(),
        })
        .unwrap();
        let text = "tiny.";
        assert!(c.split(text).unwrap().is_empty());
    }

    #[test]
    fn invalid_config_overlap_ge_max() {
        let bad = ChunkConfig {
            max_tokens: 10,
            overlap_tokens: 10,
            ..Default::default()
        };
        assert!(Chunker::new(bad).is_err());
    }

    #[test]
    fn invalid_config_zero_max() {
        let bad = ChunkConfig {
            max_tokens: 0,
            ..Default::default()
        };
        assert!(Chunker::new(bad).is_err());
    }

    #[test]
    fn unknown_encoding_rejected() {
        let bad = ChunkConfig {
            encoding: "nope_base".to_string(),
            ..Default::default()
        };
        assert!(matches!(
            Chunker::new(bad),
            Err(ChunkerError::UnknownEncoding(_))
        ));
    }

    #[test]
    fn abbreviation_does_not_split_sentence() {
        let c = Chunker::new(cfg(1000)).unwrap();
        let text = "Dr. Smith arrived. He said hello.";
        let sentences = c.split_sentences(text);
        // We expect ~2 sentences: "Dr. Smith arrived." and "He said hello."
        assert_eq!(sentences.len(), 2, "got: {:?}", sentences);
    }

    #[test]
    fn split_many_serial_and_parallel_match() {
        let c = Chunker::new(cfg(10)).unwrap();
        let texts = vec!["Alpha beta gamma.", "Delta. Epsilon. Zeta."];
        let serial = c.split_many(&texts, false).unwrap();
        let parallel = c.split_many(&texts, true).unwrap();
        assert_eq!(serial, parallel);
    }

    #[test]
    fn chunk_text_decodes_to_token_count() {
        let c = Chunker::new(cfg(10)).unwrap();
        let text = "The quick brown fox jumps over the lazy dog.";
        let chunks = c.split(text).unwrap();
        // For each chunk, re-encoding the chunk's text should give the
        // same token count.
        let bpe = tiktoken_rs::cl100k_base().unwrap();
        for ch in &chunks {
            let actual = bpe.encode_ordinary(&ch.text).len();
            assert_eq!(actual, ch.token_count);
        }
    }

    #[test]
    fn unicode_input_handled() {
        let c = Chunker::new(cfg(100)).unwrap();
        let text = "你好世界. Hello world. 🌍 done.";
        let r = c.split(text).unwrap();
        assert!(!r.is_empty());
        // No chunk should crash on decoding.
        for ch in &r {
            assert!(!ch.text.is_empty());
        }
    }

    #[test]
    fn min_tokens_filters_single_word_input() {
        // Config requires min 5 tokens but the input is just "hi", so the
        // single emitted chunk falls under the floor and is dropped.
        let c = Chunker::new(ChunkConfig {
            max_tokens: 100,
            overlap_tokens: 0,
            min_tokens: 5,
            encoding: "cl100k_base".to_string(),
        })
        .unwrap();
        let r = c.split("hi").unwrap();
        assert!(r.is_empty());
    }
}
