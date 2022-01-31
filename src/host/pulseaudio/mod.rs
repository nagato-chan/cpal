//! PulseAudio Implementation for CPAL
//! Note that this implementation is a warpper around threaded mainloop.
extern crate pulseaudio as pulse;

use std::{
    cell::RefCell,
    convert::TryInto,
    fmt::Debug,
    io::{Read, Write},
    ops::{Add, Deref},
    rc::Rc,
};

use self::pulse::callbacks::ListResult;
use self::pulse::context::introspect::SinkInfo;
use self::pulse::context::{Context, FlagSet as ContextFlagSet};
use self::pulse::format::Info;
use self::pulse::mainloop::threaded::Mainloop;
use self::pulse::proplist::Proplist;
use self::pulse::sample::{Format as PAFormat, Spec};

use self::pulse::stream::{FlagSet as StreamFlagSet, Stream as AudioStream};
use crate::{
    BuildStreamError, Data, DefaultStreamConfigError, DeviceNameError, DevicesError,
    InputCallbackInfo, OutputCallbackInfo, PauseStreamError, PlayStreamError, SampleFormat,
    SampleRate, StreamConfig, StreamError, SupportedBufferSize, SupportedStreamConfig,
    SupportedStreamConfigRange, SupportedStreamConfigsError,
};

use traits::{DeviceTrait, HostTrait, StreamTrait};

mod mainloop;

// We should get the device list by context.
pub type Devices = std::vec::IntoIter<Device>;

pub type DeviceType = self::pulse::def::Device;

#[derive(Clone)]
pub struct PARuntime {
    pub mainloop: Rc<RefCell<Mainloop>>,
    pub context: Rc<RefCell<Context>>,
}

// Definition of PulseAudio Device
//
// The Device struct also contain a standalone PulseAudio runtime (Mainloop and Context)
#[derive(Clone)]
pub struct Device {
    name: String,
    device_type: DeviceType,
    supported_configs: Rc<RefCell<Vec<SupportedStreamConfigRange>>>,
    pub mainloop: Rc<RefCell<Mainloop>>,
    pub context: Rc<RefCell<Context>>,
    // stream: Option<Rc<RefCell<AudioStream>>>,
}

impl Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("name", &self.name)
            .field("device_type", &self.device_type)
            .field("supported_config", &self.supported_configs)
            .finish()
    }
}

impl Device {
    fn get_supported_configs(&self) -> Vec<SupportedStreamConfigRange> {
        self.supported_configs.borrow().to_vec()
    }
    fn default_config(&self) -> Result<SupportedStreamConfig, DefaultStreamConfigError> {
        let mut formats = self.get_supported_configs();
        formats.sort_by(|a, b| a.cmp_default_heuristics(b));
        match formats.into_iter().last() {
            Some(f) => {
                let min_r = f.min_sample_rate;
                let max_r = f.max_sample_rate;
                let mut format = f.with_max_sample_rate();
                const HZ_44100: SampleRate = SampleRate(44_100);
                if min_r <= HZ_44100 && HZ_44100 <= max_r {
                    format.sample_rate = HZ_44100;
                }
                Ok(format)
            }
            None => Err(DefaultStreamConfigError::StreamTypeNotSupported),
        }
    }
}
/// PulseAudio Host, which is used to query data from PulseAudio server
///
/// The Host mainloop is separated from Device mainloop. It's not designed to be used intented in Device.
///
/// When the host is terminated, the context and mainloop will drop.
pub struct Host {
    // The threaded mainloop.
    pub mainloop: Rc<RefCell<Mainloop>>,
    pub context: Rc<RefCell<Context>>,
}

// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Stream {
    inner: AudioStream,
}

// Referred to PulseAudio::sample::Spec
pub type SupportedInputConfigs = std::vec::IntoIter<SupportedStreamConfigRange>;
pub type SupportedOutputConfigs = std::vec::IntoIter<SupportedStreamConfigRange>;
// Referred to PulseAudio::sample::Spec
fn parse_info(item: &Info) -> SupportedStreamConfigRange {
    // let mut config= SupportedStreamConfigRange;

    let sample_format = match item
        .get_sample_format()
        .expect("Failed to get sample format.")
    {
        Format::FLOAT32NE => Some(SampleFormat::F32),
        Format::S32NE => Some(SampleFormat::U16),
        Format::S16NE => Some(SampleFormat::I16),
        _ => None,
    }
    .expect("No support format");
    let channels = item.get_channel_count().unwrap() as u16;
    // Current I couldn't find a suitable api to implement buffer_size so i left it to unknown
    let buffer_size = SupportedBufferSize::Unknown;
    // current it's hard to find an api to query those props. so leave it the same here.
    // TODO: Fix getting min_sample_rate
    let min_sample_rate = SampleRate(item.get_rate().expect("Failed to get sample rate"));
    let max_sample_rate = SampleRate(item.get_rate().expect("Failed to get sample rate"));
    SupportedStreamConfigRange {
        channels,
        min_sample_rate,
        max_sample_rate,
        buffer_size,
        sample_format,
    }
}

// Initialize PulseAudio Runtime
// It has checked the context status.
fn setup_pa_runtime() -> (Rc<RefCell<Mainloop>>, Rc<RefCell<Context>>) {
    let mut mainloop = Rc::new(RefCell::new(
        Mainloop::new().expect("Failed to create mainloop"),
    ));
    let mut proplist = Proplist::new().expect("Failed to initialize Proplist");
    proplist
        .set_str(
            self::pulse::proplist::properties::APPLICATION_NAME,
            "CPAL_App",
        )
        .expect("Failed to set proplist");
    let mut context = Rc::new(RefCell::new(
        Context::new_with_proplist(mainloop.borrow().deref(), "CPAL_Context", &proplist)
            .expect("Failed to create context"),
    ));

    // Check context status
    {
        let ml_ref = Rc::clone(&mainloop);
        let context_ref = Rc::clone(&context);
        context
            .borrow_mut()
            .set_state_callback(Some(Box::new(move || {
                let state = unsafe { (*context_ref.as_ptr()).get_state() };
                match state {
                    self::pulse::context::State::Ready
                    | self::pulse::context::State::Failed
                    | self::pulse::context::State::Terminated => unsafe {
                        (*ml_ref.as_ptr()).signal(false);
                    },
                    _ => {}
                }
            })));
    }

    // Setup Context and Mainloop
    mainloop
        .borrow_mut()
        .start()
        .expect("Failed to start mainloop");
    mainloop.borrow_mut().lock();
    context
        .borrow_mut()
        .connect(None, ContextFlagSet::NOFLAGS, None)
        .expect("Failed to connect context"); // Query context status
    loop {
        // println!("{:?}", self.context.borrow().get_state());
        match context.borrow().get_state() {
            pulse::context::State::Ready => {
                break;
            }
            pulse::context::State::Failed | pulse::context::State::Terminated => {
                eprintln!("Context state failed/terminated, quitting...");
                mainloop.borrow_mut().unlock();
                mainloop.borrow_mut().stop();
                // return;
            }
            _ => {
                mainloop.borrow_mut().wait();
            }
        }
    }
    context.borrow_mut().set_state_callback(None);
    mainloop.borrow_mut().unlock();
    (mainloop, context)
}

impl Host {
    // Initialize the connection to PulseAudio Host
    // Should be called first
    pub fn new() -> Result<Self, crate::HostUnavailable> {
        let (mainloop, context) = setup_pa_runtime();
        let host = Host { mainloop, context };
        Ok(host)
    }
    /// Note: Be awared to use the function, it's not tend to be threat safe.
    fn get_output_device_list(&self) -> Vec<Device> {
        let devices = Rc::new(RefCell::new(vec![]));
        let dev_ref = devices.clone();
        let ml_ref = Rc::clone(&self.mainloop);
        // Lock the mainloop thread.
        self.mainloop.borrow_mut().lock();

        // self.context.borrow_mut().set_state_callback(None);
        let result = Rc::new(RefCell::new(false));
        let result_ref = result.clone();

        let mut op = self
            .context
            .borrow_mut()
            .introspect()
            .get_sink_info_list(move |sink_list| match sink_list {
                ListResult::Item(item) => {
                    let mut configs = vec![];

                    let config = SupportedStreamConfigRange {
                        channels: item.channel_map.len() as u16,
                        min_sample_rate: SampleRate(item.sample_spec.rate),
                        max_sample_rate: SampleRate(item.sample_spec.rate),
                        buffer_size: SupportedBufferSize::Unknown,
                        sample_format: item.sample_spec.format.into(),
                    };
                    configs.push(config);

                    let (mainloop, context) = setup_pa_runtime();
                    let device = Device {
                        name: item.name.as_ref().unwrap().to_string(),
                        device_type: DeviceType::Sink,
                        supported_configs: Rc::new(RefCell::new(configs)),
                        mainloop,
                        context, // stream: None,
                    };

                    dev_ref.borrow_mut().push(device)
                }
                ListResult::End => {
                    unsafe {
                        (*ml_ref.as_ptr()).signal(false);
                    }
                    *result_ref.borrow_mut() = true;
                }
                ListResult::Error => todo!(),
            });

        self.mainloop.borrow_mut().wait();

        op.cancel();

        self.mainloop.borrow_mut().unlock();
        // println!("{:#?}", self.context.borrow().get_state());
        // println!("Finish Query");

        devices.take()
    }

    /// Note: Be awared to use the function, it's not tend to be threat safe.
    fn get_input_device_list(&self) -> Vec<Device> {
        let devices = Rc::new(RefCell::new(vec![]));
        let dev_ref = devices.clone();
        let ml_ref = Rc::clone(&self.mainloop);
        // Lock the mainloop thread.
        self.mainloop.borrow_mut().lock();

        // self.context.borrow_mut().set_state_callback(None);
        let result = Rc::new(RefCell::new(false));
        let result_ref = result.clone();

        let mut op =
            self.context
                .borrow_mut()
                .introspect()
                .get_source_info_list(move |sink_list| match sink_list {
                    ListResult::Item(item) => {
                        let mut configs = vec![];

                        // FIXME:current due to the limit of pulseaudio, the config is fixed. It should be fix somewhen.
                        let config = SupportedStreamConfigRange {
                            channels: item.channel_map.len() as u16,
                            min_sample_rate: SampleRate(item.sample_spec.rate),
                            max_sample_rate: SampleRate(item.sample_spec.rate),
                            buffer_size: SupportedBufferSize::Unknown,
                            sample_format: item.sample_spec.format.into(),
                        };
                        configs.push(config);
                        let (mainloop, context) = setup_pa_runtime();
                        let device = Device {
                            name: item.name.as_ref().unwrap().to_string(),
                            device_type: DeviceType::Source,
                            supported_configs: Rc::new(RefCell::new(configs)),
                            mainloop,
                            context,
                        };

                        dev_ref.borrow_mut().push(device)
                    }
                    ListResult::End => {
                        unsafe {
                            (*ml_ref.as_ptr()).signal(false);
                        }
                        *result_ref.borrow_mut() = true;
                    }
                    ListResult::Error => todo!(),
                });

        self.mainloop.borrow_mut().wait();

        op.cancel();

        self.mainloop.borrow_mut().unlock();
        // println!("{:#?}", self.context.borrow().get_state());
        // println!("Finish Query");

        devices.take()
    }
}
impl Stream {
    fn new_input() -> Stream {
        todo!()
    }
    fn new_output() -> Stream {
        todo!()
    }
}
// NOTE: In PulseAudio, the input device is represented to source.
// and output device are represented to sink.
// It's a little bit hard to combine those two into a single device.
impl DeviceTrait for Device {
    type SupportedInputConfigs = SupportedInputConfigs;
    type SupportedOutputConfigs = SupportedOutputConfigs;
    type Stream = Stream;

    #[inline]
    // At the moment we just give a default name for convenience.
    fn name(&self) -> Result<String, DeviceNameError> {
        Ok(self.name.to_owned())
    }

    #[inline]
    fn supported_input_configs(
        &self,
    ) -> Result<SupportedInputConfigs, SupportedStreamConfigsError> {
        if self.device_type == DeviceType::Source {
            Ok(self.get_supported_configs().clone().into_iter())
        } else {
            Err(SupportedStreamConfigsError::InvalidArgument)
        }
    }

    #[inline]
    fn supported_output_configs(
        &self,
    ) -> Result<SupportedOutputConfigs, SupportedStreamConfigsError> {
        if self.device_type == DeviceType::Sink {
            Ok(self.get_supported_configs().clone().into_iter())
        } else {
            Err(SupportedStreamConfigsError::InvalidArgument)
        }
    }

    #[inline]
    fn default_input_config(&self) -> Result<SupportedStreamConfig, DefaultStreamConfigError> {
        // unimplemented!()
        if self.device_type == DeviceType::Source {
            self.default_config()
        } else {
            Err(DefaultStreamConfigError::DeviceNotAvailable)
        }
    }

    #[inline]
    fn default_output_config(&self) -> Result<SupportedStreamConfig, DefaultStreamConfigError> {
        // unimplemented!()
        if self.device_type == DeviceType::Sink {
            self.default_config()
        } else {
            Err(DefaultStreamConfigError::DeviceNotAvailable)
        }
    }

    fn build_input_stream_raw<D, E>(
        &self,
        config: &StreamConfig,
        sample_format: SampleFormat,
        data_callback: D,
        error_callback: E,
    ) -> Result<Self::Stream, BuildStreamError>
    where
        D: FnMut(&Data, &InputCallbackInfo) + Send + 'static,
        E: FnMut(StreamError) + Send + 'static,
    {
        let (mainloop, context) = setup_pa_runtime();

        unimplemented!()
    }

    /// Create an output stream.
    fn build_output_stream_raw<D, E>(
        &self,
        config: &StreamConfig,
        sample_format: SampleFormat,
        data_callback: D,
        error_callback: E,
    ) -> Result<Self::Stream, BuildStreamError>
    where
        D: FnMut(&mut Data, &OutputCallbackInfo) + Send + 'static,
        E: FnMut(StreamError) + Send + 'static,
    {
        let spec = Spec {
            format: sample_format.into(),
            rate: config.sample_rate.0,
            channels: config.channels as u8,
        };

        let stream = init_stream(
            self.mainloop.clone(),
            self.context.clone(),
            self.device_type,
            &spec,
            Some(self.name.as_str()),
        );

        loop {
            self.mainloop.borrow_mut().lock();

            let audio_mem = match stream.borrow_mut().begin_write(None).unwrap() {
                Some(data) => data,
                // If None returned, we will wait until the host returned a writable memory.
                None => {
                    self.mainloop.borrow_mut().unlock();
                    continue;
                }
            };

            let data = audio_mem.as_mut_ptr() as *mut ();
            let len = audio_mem.len() / sample_format.sample_size();
            let mut data = unsafe { Data::from_parts(data, len, sample_format) };

            // The latency info callback
            let callback_timeinfo = stream
                .borrow_mut()
                .get_timing_info()
                .expect("Failed to get timing info");
            // The buffer usec, see PulseAudio::def::TimingInfo for detail info.
            let buffer_usec = spec.bytes_to_usec(
                (callback_timeinfo.write_index - callback_timeinfo.read_index)
                    .try_into()
                    .unwrap(),
            );
            // TODO
            let callback_time =
                callback_timeinfo.sink_usec + callback_timeinfo.transport_usec + buffer_usec;
            let callback = crate::StreamInstant::from_secs_f64(callback_time.as_secs_f64());

            let playback_time = frames_to_duration(
                stream.borrow_mut().get_buffer_attr().unwrap().tlength as usize,
                config.sample_rate,
            );
            let playback = callback
                .add(playback_time)
                .expect("`playback` occurs beyond representation supported by `StreamInstant`");
            let timestamp = crate::OutputStreamTimestamp { callback, playback };
            let info = crate::OutputCallbackInfo { timestamp };

            // data_callback(&mut data, &info);
            self.mainloop.borrow_mut().unlock();
        }
        unimplemented!()
    }
}

// Convert the given duration in frames at the given sample rate to a `std::time::Duration`.
fn frames_to_duration(frames: usize, rate: crate::SampleRate) -> std::time::Duration {
    let secsf = frames as f64 / rate.0 as f64;
    let secs = secsf as u64;
    let nanos = ((secsf - secs as f64) * 1_000_000_000.0) as u32;
    std::time::Duration::new(secs, nanos)
}

type Format = PAFormat;

impl From<SampleFormat> for Format {
    fn from(sample_format: SampleFormat) -> Self {
        match sample_format {
            SampleFormat::F32 => Format::FLOAT32NE,
            SampleFormat::U16 => Format::S32NE,
            SampleFormat::I16 => Format::S16NE,
        }
    }
}

impl Into<SampleFormat> for Format {
    fn into(self) -> SampleFormat {
        match self {
            Format::FLOAT32NE => Some(SampleFormat::F32),
            Format::S32NE => Some(SampleFormat::U16),
            Format::S16NE => Some(SampleFormat::I16),
            _ => None,
        }
        .unwrap()
    }
}

// Initial a stream
//
// It will be returned when the stream is ready.
//
// The Mainloop is lock while the stream initialization, free when the stream is ready.
//
// During the initializing, it will connect to the context according to the `DeviceType`.
// (e.g. DeviceType::Sink will connect to playback)
fn init_stream(
    m: Rc<RefCell<Mainloop>>,
    ctx: Rc<RefCell<Context>>,
    dtype: DeviceType,
    spec: &Spec,
    name: Option<&str>,
) -> Rc<RefCell<AudioStream>> {
    // Lock the thread. It's important!
    m.borrow_mut().lock();
    // Initial stream by DeviceType
    let mut stream = match dtype {
        DeviceType::Source => Rc::new(RefCell::new(
            AudioStream::new(&mut ctx.borrow_mut(), "CPAL_Stream_In", &spec, None)
                .expect("Failed to create new stream"),
        )),
        DeviceType::Sink => Rc::new(RefCell::new(
            AudioStream::new(&mut ctx.borrow_mut(), "CPAL_Stream_Out", spec, None)
                .expect("Failed to create new stream"),
        )),
    };
    // set stream status callback
    {
        let ml_ref = Rc::clone(&m);
        let stream_ref = Rc::clone(&stream);
        stream
            .borrow_mut()
            .set_state_callback(Some(Box::new(move || {
                let state = unsafe { (*stream_ref.as_ptr()).get_state() };
                match state {
                    pulse::stream::State::Ready
                    | pulse::stream::State::Failed
                    | pulse::stream::State::Terminated => unsafe {
                        (*ml_ref.as_ptr()).signal(false);
                    },
                    _ => {}
                }
            })));
    }
    // connect to the host using default config.
    match dtype {
        DeviceType::Sink => {
            stream
                .borrow_mut()
                .connect_playback(name, None, StreamFlagSet::START_CORKED, None, None)
                .expect("Failed to connect record");
        }
        DeviceType::Source => {
            stream
                .borrow_mut()
                .connect_record(name, None, StreamFlagSet::START_CORKED)
                .expect("failed to connect playback");
        }
    }
    // Wait for stream to be ready
    loop {
        match stream.borrow().get_state() {
            pulse::stream::State::Ready => {
                break;
            }
            pulse::stream::State::Failed | pulse::stream::State::Terminated => {
                eprintln!("Stream state failed/terminated, quitting...");
                m.borrow_mut().unlock();
                m.borrow_mut().stop();
            }
            _ => {
                m.borrow_mut().wait();
            }
        }
    }
    // set the stream callback to None.
    stream.borrow_mut().set_state_callback(None);
    // Unlock the thread
    m.borrow_mut().unlock();

    stream
}
// The mainloop here is just for query data. so it's required to clean it.
impl Drop for Host {
    fn drop(&mut self) {
        self.mainloop.borrow_mut().lock();
        self.context.borrow_mut().disconnect();
        self.mainloop.borrow_mut().unlock();
        self.mainloop.borrow_mut().stop();
    }
}
impl HostTrait for Host {
    type Devices = Devices;
    type Device = Device;

    /// We assume the PulseAudio Backend should be always available.
    fn is_available() -> bool {
        true
    }

    fn devices(&self) -> Result<Self::Devices, DevicesError> {
        let mut device_list = vec![];
        let mut input = self.get_input_device_list();
        let mut output = self.get_output_device_list();
        device_list.append(&mut input);
        device_list.append(&mut output);
        println!("{:#?}", device_list);
        Ok(device_list.into_iter())
    }

    fn default_input_device(&self) -> Option<Device> {
        // Note: This spec is used for requesting the default device.
        let ml_ref = self.mainloop.clone();
        let ctx_ref = self.context.clone();
        let spec = Spec {
            format: Format::S16NE,
            channels: 2,
            rate: 44100,
        };
        let stream = init_stream(ml_ref, ctx_ref, DeviceType::Source, &spec, None);

        let mut stream = stream.borrow_mut();
        let sample_spec = stream.get_sample_spec().unwrap();
        let config = SupportedStreamConfigRange {
            channels: sample_spec.channels as u16,
            min_sample_rate: SampleRate(sample_spec.rate),
            max_sample_rate: SampleRate(sample_spec.rate),
            buffer_size: SupportedBufferSize::Unknown,
            sample_format: sample_spec.format.into(),
        };

        let (mainloop, context) = setup_pa_runtime();
        let default_device = Device {
            name: stream.get_device_name().unwrap().as_ref().to_string(),
            device_type: DeviceType::Source,
            supported_configs: Rc::new(RefCell::new(vec![config])),
            mainloop,
            context,
        };
        stream.disconnect().expect("Failed to drop stream");
        Some(default_device)
        // None
        // self.context.borrow().introspect()
    }

    fn default_output_device(&self) -> Option<Device> {
        // Note: This spec is used for requesting the default device.
        let ml_ref = self.mainloop.clone();
        let ctx_ref = self.context.clone();

        let spec = Spec {
            format: Format::S16NE,
            channels: 2,
            rate: 44100,
        };

        let stream = init_stream(ml_ref, ctx_ref, DeviceType::Sink, &spec, None);

        let mut stream = stream.borrow_mut();
        let sample_spec = stream.get_sample_spec().unwrap();
        let config = SupportedStreamConfigRange {
            channels: sample_spec.channels as u16,
            min_sample_rate: SampleRate(sample_spec.rate),
            max_sample_rate: SampleRate(sample_spec.rate),
            buffer_size: SupportedBufferSize::Unknown,
            sample_format: sample_spec.format.into(),
        };
        let (mainloop, context) = setup_pa_runtime();
        let default_device = Device {
            name: stream.get_device_name().unwrap().as_ref().to_string(),
            device_type: DeviceType::Source,
            supported_configs: Rc::new(RefCell::new(vec![config])),
            mainloop,
            context,
        };
        stream.disconnect().expect("Failed to drop stream");
        Some(default_device)
    }
}

impl StreamTrait for Stream {
    fn play(&self) -> Result<(), PlayStreamError> {
        unimplemented!()
    }

    fn pause(&self) -> Result<(), PauseStreamError> {
        unimplemented!()
    }
}

// impl Iterator for Devices {
//     type Item = Device;

//     #[inline]
//     fn next(&mut self) -> Option<Device> {
//         None
//     }
// }

// impl Iterator for SupportedInputConfigs {
//     type Item = SupportedStreamConfigRange;

//     #[inline]
//     fn next(&mut self) -> Option<SupportedStreamConfigRange> {
//         None
//     }
// }

// impl Iterator for SupportedOutputConfigs {
//     type Item = SupportedStreamConfigRange;

//     #[inline]
//     fn next(&mut self) -> Option<SupportedStreamConfigRange> {
//         None
//     }
// }
