//! PulseAudio Implementation for CPAL
//! Note that this implementation is a warpper around threaded mainloop.
extern crate pulseaudio as pulse;

use std::{
    cell::RefCell,
    io::Read,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use self::pulse::callbacks::ListResult;
use self::pulse::context::introspect::SinkInfo;
use self::pulse::context::{Context, FlagSet as ContextFlagSet};
use self::pulse::format::Info;
use self::pulse::mainloop::threaded::Mainloop;
use self::pulse::proplist::Proplist;
use self::pulse::sample::Format;
use self::pulse::stream::Stream as AudioStream;
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
// Definition of PulseAudio Device
#[derive(Clone, Debug)]
pub struct Device {
    name: String,
    device_type: DeviceType,
    supported_configs: Rc<RefCell<Vec<SupportedStreamConfigRange>>>,
    // stream: Option<Rc<RefCell<AudioStream>>>,
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
/// In PulseAudo, Mainloop will be referred to the host.
pub struct Host {
    // The threaded mainloop.
    pub mainloop: Rc<RefCell<Mainloop>>,
    pub context: Rc<RefCell<Context>>,
}

// Referred to PulseAudio::stream
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Stream;

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

impl Host {
    // Initialize the connection to PulseAudio Host
    // Should be called first
    pub fn new() -> Result<Self, crate::HostUnavailable> {
        let mut mainloop = Rc::new(RefCell::new(
            Mainloop::new().expect("Failed to initialize mainloop"),
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
        context
            .borrow_mut()
            .connect(None, ContextFlagSet::NOFLAGS, None)
            .expect("Failed to connect context");
        mainloop
            .borrow_mut()
            .start()
            .expect("Failed to start mainloop");
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

        // Query context status
        loop {
            // println!("{:?}", self.context.borrow().get_state());
            match self.context.borrow().get_state() {
                pulse::context::State::Ready => {
                    break;
                }
                pulse::context::State::Failed | pulse::context::State::Terminated => {
                    eprintln!("Context state failed/terminated, quitting...");
                    self.mainloop.borrow_mut().unlock();
                    self.mainloop.borrow_mut().stop();
                    // return;
                }
                _ => {
                    self.mainloop.borrow_mut().wait();
                }
            }
        }

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

                    // FIXME:current due to the limit of pulseaudio, the config is fixed. It should be fix somewhen.
                    let config = SupportedStreamConfigRange {
                        channels: item.channel_map.len() as u16,
                        min_sample_rate: SampleRate(item.sample_spec.rate),
                        max_sample_rate: SampleRate(item.sample_spec.rate),
                        buffer_size: SupportedBufferSize::Unknown,
                        sample_format: match item.sample_spec.format {
                            Format::FLOAT32NE => Some(SampleFormat::F32),
                            Format::S32NE => Some(SampleFormat::U16),
                            Format::S16NE => Some(SampleFormat::I16),
                            _ => None,
                        }
                        .expect("Unsupported format"),
                    };
                    configs.push(config);
                    let device = Device {
                        name: item.name.as_ref().unwrap().to_string(),
                        device_type: DeviceType::Sink,
                        supported_configs: Rc::new(RefCell::new(configs)),
                        // stream: None,
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

        // Query context status
        loop {
            // println!("{:?}", self.context.borrow().get_state());
            match self.context.borrow().get_state() {
                pulse::context::State::Ready => {
                    break;
                }
                pulse::context::State::Failed | pulse::context::State::Terminated => {
                    eprintln!("Context state failed/terminated, quitting...");
                    self.mainloop.borrow_mut().unlock();
                    self.mainloop.borrow_mut().stop();
                    // return;
                }
                _ => {
                    self.mainloop.borrow_mut().wait();
                }
            }
        }

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
                            sample_format: match item.sample_spec.format {
                                Format::FLOAT32NE => Some(SampleFormat::F32),
                                Format::S32NE => Some(SampleFormat::U16),
                                Format::S16NE => Some(SampleFormat::I16),
                                _ => None,
                            }
                            .expect("Unsupported format"),
                        };
                        configs.push(config);

                        let device = Device {
                            name: item.name.as_ref().unwrap().to_string(),
                            device_type: DeviceType::Source,
                            supported_configs: Rc::new(RefCell::new(configs)),
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

// impl Devices {
//     // Referred to PulseAudio::context::introspect
//     // Should be called after setting up the connection from Host.
//     pub fn new() -> Result<Self, DevicesError> {
//         Ok(Devices)
//     }
// }

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
        _config: &StreamConfig,
        _sample_format: SampleFormat,
        _data_callback: D,
        _error_callback: E,
    ) -> Result<Self::Stream, BuildStreamError>
    where
        D: FnMut(&Data, &InputCallbackInfo) + Send + 'static,
        E: FnMut(StreamError) + Send + 'static,
    {
        unimplemented!()
    }

    /// Create an output stream.
    fn build_output_stream_raw<D, E>(
        &self,
        _config: &StreamConfig,
        _sample_format: SampleFormat,
        _data_callback: D,
        _error_callback: E,
    ) -> Result<Self::Stream, BuildStreamError>
    where
        D: FnMut(&mut Data, &OutputCallbackInfo) + Send + 'static,
        E: FnMut(StreamError) + Send + 'static,
    {
        unimplemented!()
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
        // todo!()
    }

    fn default_input_device(&self) -> Option<Device> {
        None
        // self.context.borrow().introspect()
    }

    fn default_output_device(&self) -> Option<Device> {
        None
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
