#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Addr(pub(crate) u64);

impl std::fmt::Display for Addr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Addr {
    pub fn offset(self, x: u64) -> Addr {
        Addr(self.0 + x)
    }

    pub fn unoffset(self, x: u64) -> Addr {
        Addr(self.0 - x)
    }

    pub fn get(self) -> usize {
        self.0 as usize
    }
}
