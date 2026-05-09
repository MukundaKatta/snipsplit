# snipsplit-core

Pure-Rust core for [snipsplit](https://github.com/MukundaKatta/snipsplit):
a token-aware text chunker for RAG ingestion.

```rust
use snipsplit_core::{ChunkConfig, Chunker};

let chunker = Chunker::new(ChunkConfig {
    max_tokens: 256,
    overlap_tokens: 32,
    ..Default::default()
})?;
let chunks = chunker.split("Some long document...")?;
for c in &chunks {
    println!("{}..{} ({} tokens)", c.start, c.end, c.token_count);
}
# Ok::<(), snipsplit_core::ChunkerError>(())
```

## License

Dual-licensed under MIT or Apache-2.0.
