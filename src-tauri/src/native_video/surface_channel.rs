use std::io;
use std::mem;
use std::os::fd::{FromRawFd, OwnedFd, RawFd};

use serde::{Deserialize, Serialize};

use super::surface_protocol::{SurfaceDescriptor, SurfaceFrameReady, SurfaceFrameRelease};

const MAX_MESSAGE_BYTES: usize = 64 * 1024;
const MAX_MESSAGE_FDS: usize = 5;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SurfaceChannelMessage {
    SurfaceCreated { descriptor: SurfaceDescriptor },
    FrameReady { frame: SurfaceFrameReady },
    FrameRelease { release: SurfaceFrameRelease },
}

#[derive(Debug)]
pub struct ReceivedSurfaceMessage {
    pub message: SurfaceChannelMessage,
    pub fds: Vec<OwnedFd>,
}

pub fn socket_pair() -> io::Result<(OwnedFd, OwnedFd)> {
    let mut sockets = [-1; 2];
    let result = unsafe {
        libc::socketpair(
            libc::AF_UNIX,
            libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC,
            0,
            sockets.as_mut_ptr(),
        )
    };
    if result != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe {
        (
            OwnedFd::from_raw_fd(sockets[0]),
            OwnedFd::from_raw_fd(sockets[1]),
        )
    })
}

pub fn send_message(
    socket: RawFd,
    message: &SurfaceChannelMessage,
    fds: &[RawFd],
) -> io::Result<()> {
    if fds.len() > MAX_MESSAGE_FDS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "surface message has too many file descriptors",
        ));
    }
    let payload = serde_json::to_vec(message)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    if payload.len() > MAX_MESSAGE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "surface message exceeds the bounded payload size",
        ));
    }

    let mut iov = libc::iovec {
        iov_base: payload.as_ptr().cast_mut().cast(),
        iov_len: payload.len(),
    };
    let control_bytes = if fds.is_empty() {
        0
    } else {
        unsafe { libc::CMSG_SPACE(mem::size_of_val(fds) as u32) as usize }
    };
    let mut control = vec![0_u8; control_bytes];
    let mut header: libc::msghdr = unsafe { mem::zeroed() };
    header.msg_iov = &mut iov;
    header.msg_iovlen = 1;
    if !control.is_empty() {
        header.msg_control = control.as_mut_ptr().cast();
        header.msg_controllen = control.len();
        unsafe {
            let cmsg = libc::CMSG_FIRSTHDR(&header);
            (*cmsg).cmsg_level = libc::SOL_SOCKET;
            (*cmsg).cmsg_type = libc::SCM_RIGHTS;
            (*cmsg).cmsg_len = libc::CMSG_LEN(mem::size_of_val(fds) as u32) as usize;
            std::ptr::copy_nonoverlapping(
                fds.as_ptr().cast::<u8>(),
                libc::CMSG_DATA(cmsg),
                mem::size_of_val(fds),
            );
        }
    }

    let sent = unsafe { libc::sendmsg(socket, &header, libc::MSG_NOSIGNAL) };
    if sent < 0 {
        return Err(io::Error::last_os_error());
    }
    if sent as usize != payload.len() {
        return Err(io::Error::new(
            io::ErrorKind::WriteZero,
            "surface channel sent a partial packet",
        ));
    }
    Ok(())
}

pub fn receive_message(socket: RawFd) -> io::Result<ReceivedSurfaceMessage> {
    let mut payload = vec![0_u8; MAX_MESSAGE_BYTES];
    let mut iov = libc::iovec {
        iov_base: payload.as_mut_ptr().cast(),
        iov_len: payload.len(),
    };
    let control_bytes =
        unsafe { libc::CMSG_SPACE((MAX_MESSAGE_FDS * mem::size_of::<RawFd>()) as u32) as usize };
    let mut control = vec![0_u8; control_bytes];
    let mut header: libc::msghdr = unsafe { mem::zeroed() };
    header.msg_iov = &mut iov;
    header.msg_iovlen = 1;
    header.msg_control = control.as_mut_ptr().cast();
    header.msg_controllen = control.len();

    let received = unsafe { libc::recvmsg(socket, &mut header, libc::MSG_CMSG_CLOEXEC) };
    if received < 0 {
        return Err(io::Error::last_os_error());
    }
    if received == 0 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "surface channel closed",
        ));
    }
    if header.msg_flags & (libc::MSG_TRUNC | libc::MSG_CTRUNC) != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "surface channel packet or descriptors were truncated",
        ));
    }
    payload.truncate(received as usize);

    let mut raw_fds = Vec::new();
    unsafe {
        let mut cmsg = libc::CMSG_FIRSTHDR(&header);
        while !cmsg.is_null() {
            if (*cmsg).cmsg_level == libc::SOL_SOCKET && (*cmsg).cmsg_type == libc::SCM_RIGHTS {
                let data_bytes = (*cmsg).cmsg_len - libc::CMSG_LEN(0) as usize;
                let count = data_bytes / mem::size_of::<RawFd>();
                let data = libc::CMSG_DATA(cmsg).cast::<RawFd>();
                for index in 0..count {
                    raw_fds.push(*data.add(index));
                }
            }
            cmsg = libc::CMSG_NXTHDR(&header, cmsg);
        }
    }
    if raw_fds.len() > MAX_MESSAGE_FDS {
        for fd in raw_fds {
            unsafe { libc::close(fd) };
        }
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "surface channel received too many descriptors",
        ));
    }

    let message = match serde_json::from_slice(&payload) {
        Ok(message) => message,
        Err(error) => {
            for fd in raw_fds {
                unsafe { libc::close(fd) };
            }
            return Err(io::Error::new(io::ErrorKind::InvalidData, error));
        }
    };
    let fds = raw_fds
        .into_iter()
        .map(|fd| unsafe { OwnedFd::from_raw_fd(fd) })
        .collect();
    Ok(ReceivedSurfaceMessage { message, fds })
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::os::fd::AsRawFd;

    use super::*;
    use crate::native_video::surface_protocol::{SurfaceHandleKind, SurfacePlane};

    #[test]
    fn seqpacket_channel_preserves_one_message_and_ancillary_fd() {
        let (sender, receiver) = socket_pair().unwrap();
        let file = File::open("/dev/null").unwrap();
        let descriptor = SurfaceDescriptor {
            generation: 4,
            buffer_id: 2,
            width: 640,
            height: 360,
            sample_aspect_ratio: 1.0,
            rotation_degrees: 0,
            fourcc: 0x3432_4241,
            modifier: 0,
            handle_kind: SurfaceHandleKind::SharedMemory,
            reusable_dmabuf: false,
            producer_drm_node: None,
            color_space: None,
            color_range: None,
            chroma_location: None,
            planes: vec![SurfacePlane {
                stride: 640 * 4,
                offset: 0,
            }],
            dmabuf: None,
        };
        let message = SurfaceChannelMessage::SurfaceCreated {
            descriptor: descriptor.clone(),
        };
        send_message(sender.as_raw_fd(), &message, &[file.as_raw_fd()]).unwrap();
        let received = receive_message(receiver.as_raw_fd()).unwrap();
        assert_eq!(received.message, message);
        assert_eq!(received.fds.len(), 1);
        assert!(descriptor.validate(received.fds.len()).is_ok());
        assert!(unsafe { libc::fcntl(received.fds[0].as_raw_fd(), libc::F_GETFD) } >= 0);
    }

    #[test]
    fn channel_rejects_more_than_the_bounded_descriptor_count() {
        let (sender, _receiver) = socket_pair().unwrap();
        let file = File::open("/dev/null").unwrap();
        let fds = vec![file.as_raw_fd(); MAX_MESSAGE_FDS + 1];
        let message = SurfaceChannelMessage::FrameRelease {
            release: SurfaceFrameRelease {
                generation: 1,
                buffer_id: 0,
                sequence: 1,
            },
        };
        assert_eq!(
            send_message(sender.as_raw_fd(), &message, &fds)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
    }
}
