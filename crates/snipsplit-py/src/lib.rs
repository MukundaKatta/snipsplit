//! PyO3 bindings exposing `snipsplit_core` as `snipsplit._native`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

use snipsplit_core::{ChunkConfig, Chunker, ChunkerError};

pyo3::create_exception!(_native, SnipsplitError, pyo3::exceptions::PyException);

fn map_err(e: ChunkerError) -> PyErr {
    match e {
        ChunkerError::InvalidConfig(_) | ChunkerError::UnknownEncoding(_) => {
            PyValueError::new_err(e.to_string())
        }
        other => SnipsplitError::new_err(other.to_string()),
    }
}

#[pyclass(name = "Chunk", module = "snipsplit._native", frozen)]
#[derive(Clone)]
struct PyChunk {
    inner: snipsplit_core::Chunk,
}

#[pymethods]
impl PyChunk {
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }
    #[getter]
    fn start(&self) -> usize {
        self.inner.start
    }
    #[getter]
    fn end(&self) -> usize {
        self.inner.end
    }
    #[getter]
    fn token_count(&self) -> usize {
        self.inner.token_count
    }
    fn __repr__(&self) -> String {
        format!(
            "Chunk(start={}, end={}, token_count={}, text={:?})",
            self.inner.start,
            self.inner.end,
            self.inner.token_count,
            preview(&self.inner.text, 40),
        )
    }
}

fn preview(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.to_string()
    } else {
        let s: String = text.chars().take(max).collect();
        format!("{s}...")
    }
}

#[pyclass(name = "Chunker", module = "snipsplit._native")]
struct PyChunker {
    inner: Chunker,
}

#[pymethods]
impl PyChunker {
    #[new]
    #[pyo3(signature = (*, max_tokens=512, overlap_tokens=0, min_tokens=1, encoding="cl100k_base", preserve_paragraphs=false))]
    fn new(
        max_tokens: usize,
        overlap_tokens: usize,
        min_tokens: usize,
        encoding: &str,
        preserve_paragraphs: bool,
    ) -> PyResult<Self> {
        let inner = Chunker::new(ChunkConfig {
            max_tokens,
            overlap_tokens,
            min_tokens,
            encoding: encoding.to_string(),
            preserve_paragraphs,
        })
        .map_err(map_err)?;
        Ok(Self { inner })
    }

    fn split(&self, py: Python<'_>, text: &str) -> PyResult<Vec<PyChunk>> {
        let owned = text.to_owned();
        let raw = py
            .allow_threads(move || self.inner.split(&owned))
            .map_err(map_err)?;
        Ok(raw.into_iter().map(|inner| PyChunk { inner }).collect())
    }

    #[pyo3(signature = (texts, parallel=false))]
    fn split_many(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        parallel: bool,
    ) -> PyResult<Vec<Vec<PyChunk>>> {
        let raw = py
            .allow_threads(move || {
                let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
                self.inner.split_many(&refs, parallel)
            })
            .map_err(map_err)?;
        Ok(raw
            .into_iter()
            .map(|v| v.into_iter().map(|inner| PyChunk { inner }).collect())
            .collect())
    }

    fn __repr__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyString>> {
        Ok(PyString::new(py, "Chunker(...)"))
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("SnipsplitError", m.py().get_type::<SnipsplitError>())?;
    m.add_class::<PyChunk>()?;
    m.add_class::<PyChunker>()?;
    Ok(())
}
