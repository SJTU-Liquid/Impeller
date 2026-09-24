use super::{Error, HardwareInfo, MemoryInfo};
use crate::utils::human_readable_size;
use std::collections::{BTreeSet, HashMap};
use std::io::Read;
use zkpoly_runtime::args::RuntimeType;

pub struct VersionsHelper<'a> {
    mem_info: &'a MemoryInfo,
}

impl<'a> VersionsHelper<'a> {
    pub fn pass_name<S>(&self, msg: S) -> String
    where
        S: AsRef<str>,
    {
        format!(
            "{} for {}",
            msg.as_ref(),
            human_readable_size(self.mem_info.memory_limit()).replace('.', "_")
        )
    }

    pub fn dirname(&self, f: &'static str) -> String {
        format!(
            "{}_{}",
            f,
            human_readable_size(self.mem_info.memory_limit()).replace('.', "_")
        )
    }
}

#[derive(Clone)]
pub struct Versions<T>(Vec<(MemoryInfo, T)>);

impl<T> Versions<T> {
    pub fn build<'s, Rt: RuntimeType>(
        hardware_info: &HardwareInfo,
        divisors: impl Iterator<Item = u32>,
        mut f: impl for<'a> FnMut(&'a MemoryInfo, VersionsHelper<'a>) -> Result<T, Error<'s, Rt>>,
    ) -> Result<Self, Error<'s, Rt>> {
        let infos = hardware_info.smaller_cpus(divisors).collect::<Vec<_>>();
        infos.iter().fold(HashMap::<String, MemoryInfo>::new(), |mut acc, e| {
            let s = human_readable_size(e.memory_limit());
            if let Some(m) = acc.get(&s) {
                panic!("We are distinguishing versions of artifect using human readable CPU memory limits, but your configuration of versions are undistinguishable: One with size {}({}) and the other with {}", e.memory_limit(), s, m.memory_limit())
            } else {
                acc.insert(s, e.clone());
                acc
            }
        });

        Ok(Self(
            infos
                .into_iter()
                .map(|cpu| {
                    let t = f(&cpu, VersionsHelper { mem_info: &cpu })?;
                    Ok((cpu, t))
                })
                .collect::<Result<_, Error<'s, Rt>>>()?,
        ))
    }

    pub fn map<'s, T2, Rt: RuntimeType>(
        self,
        mut f: impl for<'a> FnMut(&'a MemoryInfo, T, VersionsHelper<'a>) -> Result<T2, Error<'s, Rt>>,
    ) -> Result<Versions<T2>, Error<'s, Rt>> {
        Ok(Versions(
            self.0
                .into_iter()
                .map(|(cpu, t)| {
                    let t2 = f(&cpu, t, VersionsHelper { mem_info: &cpu })?;
                    Ok((cpu, t2))
                })
                .collect::<Result<_, Error<'s, Rt>>>()?,
        ))
    }

    pub fn map_i<'s, T2, Rt: RuntimeType>(
        self,
        mut f: impl for<'a> FnMut(
            &'a MemoryInfo,
            T,
            VersionsHelper<'a>,
        ) -> Result<(MemoryInfo, T2), Error<'s, Rt>>,
    ) -> Result<Versions<T2>, Error<'s, Rt>> {
        let mut existent_versions = BTreeSet::new();

        Ok(Versions(
            self.0
                .into_iter()
                .filter_map(|(cpu, t)| {
                    let (cpu, t2) = match f(&cpu, t, VersionsHelper { mem_info: &cpu }) {
                        Ok(x) => x,
                        Err(e) => return Some(Err(e)),
                    };
                    if existent_versions.contains(&cpu) {
                        None
                    } else {
                        existent_versions.insert(cpu.clone());
                        Some(Ok((cpu, t2)))
                    }
                })
                .collect::<Result<_, Error<'s, Rt>>>()?,
        ))
    }

    pub fn dump(
        &self,
        dir: impl AsRef<std::path::Path>,
        mut write: impl FnMut(std::fs::File, &MemoryInfo, &T) -> std::io::Result<()>,
    ) -> std::io::Result<()> {
        for (cpu, t) in self.0.iter() {
            let f = std::fs::File::create(dir.as_ref().join(format!(
                "version_{}.json",
                human_readable_size(cpu.memory_limit)
            )))?;

            write(f, cpu, t)?;
        }
        Ok(())
    }

    pub fn load(
        dir: impl AsRef<std::path::Path>,
        mut read: impl FnMut(std::fs::File) -> std::io::Result<(MemoryInfo, T)>,
    ) -> std::io::Result<Self> {
        let mut r = Vec::new();

        for entry in dir.as_ref().read_dir()? {
            let path = entry?.path();
            if path
                .file_stem()
                .is_some_and(|f| f.to_string_lossy().into_owned().starts_with("version_"))
                && path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.to_string_lossy().into_owned() == "json")
            {
                let f = std::fs::File::open(path)?;

                let x = read(f)?;
                r.push(x);
            }
        }
        Ok(Self(r))
    }

    pub fn load_buffered<'b>(
        dir: impl AsRef<std::path::Path>,
        buffers: &'b mut Vec<String>,
        mut read: impl FnMut(&'b str) -> std::io::Result<(MemoryInfo, T)>,
    ) -> std::io::Result<Self>
    where
        T: 'b,
    {
        let mut r = Vec::new();

        for entry in dir.as_ref().read_dir()? {
            let path = entry?.path();
            if path
                .file_stem()
                .is_some_and(|f| f.to_string_lossy().into_owned().starts_with("version_"))
                && path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.to_string_lossy().into_owned() == "json")
            {
                let mut f = std::fs::File::open(path)?;

                let mut buffer = String::new();
                f.read_to_string(&mut buffer)?;

                // SAFETY: All existing immutable reference to `buffers` are to elements before the one we are pushing here
                unsafe {
                    let buffers: *mut Vec<String> = buffers as *const _ as _;
                    buffers.as_mut().unwrap().push(buffer);
                }

                let x = read(buffers.last().unwrap())?;
                r.push(x);
            }
        }
        Ok(Self(r))
    }

    pub fn iter(&self) -> impl Iterator<Item = &MemoryInfo> {
        self.0.iter().map(|(m, _)| m)
    }

    pub fn iter_items(&self) -> impl Iterator<Item = &(MemoryInfo, T)> {
        self.0.iter()
    }

    pub fn ref_of(&self, mem: &MemoryInfo) -> Option<&T> {
        self.0.iter().find(|(m, _)| m == mem).map(|(_, t)| t)
    }
}
