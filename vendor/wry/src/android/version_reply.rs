// Copyright 2026 DonutsDelivery
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: MIT

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use std::time::Duration;

// Cold Android startup can occupy the Java main thread beyond the first probe.
// Wait for the original reply; do not queue duplicate probes or invent a version.
pub(crate) fn receive_version<E: From<RecvTimeoutError>>(
  rx: &Receiver<Result<String, E>>,
  initial_timeout: Duration,
  startup_grace: Duration,
) -> Result<String, E> {
  match rx.recv_timeout(initial_timeout) {
    Err(RecvTimeoutError::Timeout) => rx.recv_timeout(startup_grace)?,
    Err(error) => Err(error.into()),
    Ok(version) => version,
  }
}

// Shutdown can cancel the Rust receiver while Android is still answering.
pub(crate) fn send_version<E>(tx: &Sender<Result<String, E>>, version: Result<String, E>) {
  let _ = tx.send(version);
}

#[cfg(test)]
mod tests {
  use super::*;
  use crossbeam_channel::bounded;

  #[derive(Debug, PartialEq)]
  enum ProbeError {
    Channel(RecvTimeoutError),
    Java,
  }

  impl From<RecvTimeoutError> for ProbeError {
    fn from(error: RecvTimeoutError) -> Self {
      Self::Channel(error)
    }
  }

  #[test]
  fn ready_reply_preserves_exact_provider_version() {
    let (tx, rx) = bounded(1);
    send_version(&tx, Ok::<_, ProbeError>("134.0.6998.108".into()));
    assert_eq!(
      receive_version(&rx, Duration::ZERO, Duration::ZERO),
      Ok("134.0.6998.108".into())
    );
  }

  #[test]
  fn delayed_reply_survives_initial_probe_deadline() {
    let (tx, rx) = bounded(1);
    let worker = std::thread::spawn(move || {
      std::thread::sleep(Duration::from_millis(20));
      send_version(&tx, Ok::<_, ProbeError>("134.0.6998.108".into()));
    });
    assert_eq!(
      receive_version(&rx, Duration::ZERO, Duration::from_secs(5)),
      Ok("134.0.6998.108".into())
    );
    worker.join().unwrap();
  }

  #[test]
  fn exhausted_deadline_returns_timeout_and_late_reply_is_safe() {
    let (tx, rx) = bounded(1);
    assert_eq!(
      receive_version::<ProbeError>(&rx, Duration::ZERO, Duration::ZERO),
      Err(ProbeError::Channel(RecvTimeoutError::Timeout))
    );
    drop(rx);
    send_version(&tx, Ok("late version".into()));
  }

  #[test]
  fn disconnected_provider_returns_error() {
    let (tx, rx) = bounded::<Result<String, ProbeError>>(1);
    drop(tx);
    assert_eq!(
      receive_version(&rx, Duration::ZERO, Duration::from_secs(5)),
      Err(ProbeError::Channel(RecvTimeoutError::Disconnected))
    );
  }

  #[test]
  fn java_error_is_not_replaced_by_a_fake_version() {
    let (tx, rx) = bounded(1);
    send_version(&tx, Err(ProbeError::Java));
    assert_eq!(
      receive_version(&rx, Duration::ZERO, Duration::ZERO),
      Err(ProbeError::Java)
    );
  }
}
