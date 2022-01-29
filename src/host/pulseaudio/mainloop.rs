use super::pulse::context::{Context, FlagSet as ContextFlagSet};
use super::pulse::mainloop::api::Mainloop as MainloopTrait;
use super::pulse::proplist::Proplist;
use super::pulse::sample::Format;
use super::pulse::sample::Spec;
use super::pulse::stream::{FlagSet as StreamFlagSet, Stream};
use super::pulse::{context, mainloop::threaded::Mainloop, proplist::properties};
use std::{cell::RefCell, ops::Deref, rc::Rc};
pub fn init_mainloop() -> Rc<RefCell<Mainloop>> {
    // sample specification, should be defined first.
    let spec = Spec {
        format: Format::FLOAT32NE,
        channels: 2,
        rate: 44100,
    };
    // Mainloop Initialize
    let mut mainloop = Rc::new(RefCell::new(
        Mainloop::new().expect("Failed to initialize mainloop"),
    ));
    return mainloop;
    // Context Initialize
}

// fn init_context() {
//     let mut proplist = Proplist::new().expect("Failed to initialize Proplist");
//     // TODO: SHOULD Exposed to CPAL
//     proplist.set_str(properties::APPLICATION_NAME, "CPAL_App");
//     let mut context = Rc::new(RefCell::new(
//         Context::new_with_proplist(mainloop.borrow().deref(), "CPAL_Context", &proplist)
//             .expect("Failed to create context"),
//     ));
//     // Context state change callback
//     {
//         let ml_ref = Rc::clone(&mainloop);
//         let context_ref = Rc::clone(&context);
//         context
//             .borrow_mut()
//             .set_state_callback(Some(Box::new(move || {
//                 let state = unsafe { (*context_ref.as_ptr()).get_state() };
//                 match state {
//                     context::State::Ready
//                     | context::State::Failed
//                     | context::State::Terminated => unsafe {
//                         (*ml_ref.as_ptr()).signal(false);
//                     },
//                     _ => {}
//                 }
//             })));
//     }

//     context
//         .borrow_mut()
//         .connect(None, ContextFlagSet::NOFLAGS, None)
//         .expect("Failed to connect context");

//     mainloop.borrow_mut().lock();
//     mainloop
//         .borrow_mut()
//         .start()
//         .expect("Failed to start mainloop");

//     // Wait for context to be ready
//     loop {
//         match context.borrow().get_state() {
//             super::pulse::context::State::Ready => {
//                 break;
//             }
//             super::pulse::context::State::Failed | super::pulse::context::State::Terminated => {
//                 eprintln!("Context state failed/terminated, quitting...");
//                 mainloop.borrow_mut().unlock();
//                 mainloop.borrow_mut().stop();
//                 return;
//             }
//             _ => {
//                 mainloop.borrow_mut().wait();
//             }
//         }
//     }
//     context.borrow_mut().set_state_callback(None);
//     // Stream Initialize
//     let mut stream = Rc::new(RefCell::new(
//         Stream::new(&mut context.borrow_mut(), "Music", &spec, None)
//             .expect("Failed to create new stream"),
//     ));

//     // Stream state change callback
//     {
//         let ml_ref = Rc::clone(&mainloop);
//         let stream_ref = Rc::clone(&stream);
//         stream
//             .borrow_mut()
//             .set_state_callback(Some(Box::new(move || {
//                 let state = unsafe { (*stream_ref.as_ptr()).get_state() };
//                 match state {
//                     super::pulse::stream::State::Ready
//                     | super::pulse::stream::State::Failed
//                     | super::pulse::stream::State::Terminated => unsafe {
//                         (*ml_ref.as_ptr()).signal(false);
//                     },
//                     _ => {}
//                 }
//             })));
//     }

//     stream
//         .borrow_mut()
//         .connect_playback(None, None, StreamFlagSet::START_CORKED, None, None)
//         .expect("Failed to connect playback");

//     // Wait for stream to be ready
//     loop {
//         match stream.borrow().get_state() {
//             super::pulse::stream::State::Ready => {
//                 break;
//             }
//             super::pulse::stream::State::Failed | super::pulse::stream::State::Terminated => {
//                 eprintln!("Stream state failed/terminated, quitting...");
//                 mainloop.borrow_mut().unlock();
//                 mainloop.borrow_mut().stop();
//                 return;
//             }
//             _ => {
//                 mainloop.borrow_mut().wait();
//             }
//         }
//     }
//     stream.borrow_mut().set_state_callback(None);

//     mainloop.borrow_mut().unlock();

//     // Our main logic (to output a stream of audio data)
//     loop {
//         mainloop.borrow_mut().lock();

//         // Write some data with stream.write()

//         if stream.borrow().is_corked().unwrap() {
//             stream.borrow_mut().uncork(None);
//         }

//         // Drain
//         let o = {
//             let ml_ref = Rc::clone(&mainloop);
//             stream
//                 .borrow_mut()
//                 .drain(Some(Box::new(move |_success: bool| unsafe {
//                     (*ml_ref.as_ptr()).signal(false);
//                 })))
//         };
//         while o.get_state() != super::pulse::operation::State::Done {
//             mainloop.borrow_mut().wait();
//         }

//         mainloop.borrow_mut().unlock();

//         // If done writing data, call `mainloop.borrow_mut().stop()` (with lock released), then
//         // break!
//     }

//     // Clean shutdown
//     mainloop.borrow_mut().lock();
//     stream.borrow_mut().disconnect().unwrap();
//     mainloop.borrow_mut().unlock();
// }
