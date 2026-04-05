use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::fs::File;

const MAGIC: &[u8; 8] = b"C4BOOK\0\0";

/// Read-only opening book: sorted Vec of (position_key, exact_score) pairs.
/// Binary search gives O(log n) lookup with good cache locality.
/// Scores are stored as i8 (range -18..=18 fits comfortably).
pub struct OpeningBook {
    entries: Vec<(u64, i8)>, // sorted by key after finalize()
}

impl OpeningBook {
    pub fn new() -> Self {
        OpeningBook { entries: Vec::new() }
    }

    /// Push an entry during build phase. Call finalize() before any get().
    pub fn insert(&mut self, key: u64, score: i32) {
        self.entries.push((key, score as i8));
    }

    /// Sort entries and remove duplicates. Must be called after all inserts.
    pub fn finalize(&mut self) {
        self.entries.sort_unstable_by_key(|&(k, _)| k);
        self.entries.dedup_by_key(|e| e.0);
    }

    /// Merge another book's entries into this one (call finalize() afterwards).
    pub fn merge_from(&mut self, other: &OpeningBook) {
        self.entries.extend_from_slice(&other.entries);
    }

    /// Look up the exact score for a position key. O(log n) binary search.
    #[inline]
    pub fn get(&self, key: u64) -> Option<i32> {
        self.entries
            .binary_search_by_key(&key, |&(k, _)| k)
            .ok()
            .map(|idx| self.entries[idx].1 as i32)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Save to binary file.
    /// Format: magic(8) + entry_count(8 LE) + N×(key: u64 LE, score: i8)
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);
        w.write_all(MAGIC)?;
        w.write_all(&(self.entries.len() as u64).to_le_bytes())?;
        for &(key, score) in &self.entries {
            w.write_all(&key.to_le_bytes())?;
            w.write_all(&[score as u8])?;
        }
        w.flush()?;
        Ok(())
    }

    /// Load from binary file saved with save().
    pub fn load(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut r = BufReader::new(file);

        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid opening book file (bad magic bytes)",
            ));
        }

        let mut count_buf = [0u8; 8];
        r.read_exact(&mut count_buf)?;
        let count = u64::from_le_bytes(count_buf) as usize;

        let mut entries = Vec::with_capacity(count);
        let mut buf = [0u8; 9];
        for _ in 0..count {
            r.read_exact(&mut buf)?;
            let key = u64::from_le_bytes(buf[0..8].try_into().unwrap());
            let score = buf[8] as i8;
            entries.push((key, score));
        }
        // Loaded files are pre-sorted; trust the format.
        Ok(OpeningBook { entries })
    }
}
