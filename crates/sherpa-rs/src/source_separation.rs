use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::path::Path;

#[derive(Debug)]
pub struct SourceSeparation {
    ss: *const sherpa_rs_sys::SherpaOnnxOfflineSourceSeparation,
}

#[derive(Debug, Clone)]
pub struct SeparatedStem {
    pub samples: Vec<f32>,
    pub sample_rate: i32,
    pub num_channels: i32,
}

#[derive(Debug, Clone)]
pub struct SourceSeparationResult {
    pub stems: Vec<SeparatedStem>,
}

#[derive(Debug, Clone)]
pub struct SpleeterModelConfig {
    pub vocals: String,
    pub accompaniment: String,
}

#[derive(Debug, Clone)]
pub struct UvrModelConfig {
    pub model: String,
}

#[derive(Debug, Clone, Default)]
pub struct SourceSeparationConfig {
    pub spleeter: Option<SpleeterModelConfig>,
    pub uvr: Option<UvrModelConfig>,
    pub num_threads: i32,
    pub provider: Option<String>,
    pub debug: bool,
}

impl SourceSeparation {
    pub fn new_spleeter<P: AsRef<Path>>(
        vocals_model: P,
        accompaniment_model: P,
        config: SourceSeparationConfig,
    ) -> Result<Self> {
        let mut cfg = config;
        cfg.spleeter = Some(SpleeterModelConfig {
            vocals: vocals_model.as_ref().to_string_lossy().into_owned(),
            accompaniment: accompaniment_model.as_ref().to_string_lossy().into_owned(),
        });
        Self::new(cfg)
    }

    pub fn new_uvr<P: AsRef<Path>>(model: P, config: SourceSeparationConfig) -> Result<Self> {
        let mut cfg = config;
        cfg.uvr = Some(UvrModelConfig {
            model: model.as_ref().to_string_lossy().into_owned(),
        });
        Self::new(cfg)
    }

    pub fn new(config: SourceSeparationConfig) -> Result<Self> {
        let provider = config.provider.unwrap_or_else(get_default_provider);
        let debug = if config.debug { 1 } else { 0 };
        let num_threads = if config.num_threads > 0 {
            config.num_threads
        } else {
            1
        };

        let (spleeter_vocals, spleeter_accompaniment) = match &config.spleeter {
            Some(s) => (
                cstring_from_str(&s.vocals),
                cstring_from_str(&s.accompaniment),
            ),
            None => (cstring_from_str(""), cstring_from_str("")),
        };

        let uvr_model = match &config.uvr {
            Some(u) => cstring_from_str(&u.model),
            None => cstring_from_str(""),
        };

        let provider_cstr = cstring_from_str(&provider);

        let c_config = sherpa_rs_sys::SherpaOnnxOfflineSourceSeparationConfig {
            model: sherpa_rs_sys::SherpaOnnxOfflineSourceSeparationModelConfig {
                spleeter: sherpa_rs_sys::SherpaOnnxOfflineSourceSeparationSpleeterModelConfig {
                    vocals: spleeter_vocals.as_ptr(),
                    accompaniment: spleeter_accompaniment.as_ptr(),
                },
                uvr: sherpa_rs_sys::SherpaOnnxOfflineSourceSeparationUvrModelConfig {
                    model: uvr_model.as_ptr(),
                },
                num_threads,
                debug,
                provider: provider_cstr.as_ptr(),
            },
        };

        let ss =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineSourceSeparation(&c_config) };

        if ss.is_null() {
            bail!("Failed to create source separation instance");
        }

        Ok(Self { ss })
    }

    pub fn get_sample_rate(&self) -> i32 {
        unsafe { sherpa_rs_sys::SherpaOnnxOfflineSourceSeparationGetSampleRate(self.ss) }
    }

    pub fn get_num_stems(&self) -> i32 {
        unsafe { sherpa_rs_sys::SherpaOnnxOfflineSourceSeparationGetNumStems(self.ss) }
    }

    pub fn process(
        &self,
        samples: &[f32],
        sample_rate: i32,
        num_channels: i32,
    ) -> Result<SourceSeparationResult> {
        let result = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineSourceSeparationProcess(
                self.ss,
                samples.as_ptr(),
                samples.len() as i32,
                sample_rate,
                num_channels,
            )
        };

        if result.is_null() {
            bail!("Source separation processing failed");
        }

        let mut stems = Vec::new();

        unsafe {
            let num_stems = (*result).num_stems;
            let stems_ptr = (*result).stems;

            for i in 0..num_stems {
                let stem = &*stems_ptr.offset(i as isize);
                let n = stem.n as usize;
                let samples_slice = std::slice::from_raw_parts(stem.samples, n);

                stems.push(SeparatedStem {
                    samples: samples_slice.to_vec(),
                    sample_rate: stem.sample_rate,
                    num_channels: stem.num_channels,
                });
            }

            sherpa_rs_sys::SherpaOnnxDestroyOfflineSourceSeparationResult(result);
        }

        Ok(SourceSeparationResult { stems })
    }
}

unsafe impl Send for SourceSeparation {}
unsafe impl Sync for SourceSeparation {}

impl Drop for SourceSeparation {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineSourceSeparation(self.ss);
        }
    }
}
