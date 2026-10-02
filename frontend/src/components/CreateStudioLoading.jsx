export default function CreateStudioLoading({ opening, backendReady, connectionFailed }) {
  return <section className="create-studio-loading" aria-label="Studio connection" aria-busy={!connectionFailed}>
    <aside className="create-loading-sidebar" aria-hidden="true">
      <div className="create-loading-tabs"><span>Create</span><span>Edit</span><span>Finalize</span></div>
      <div className="create-loading-fields">
        <span className="create-loading-line create-loading-line-short" />
        <div className="create-loading-prompt" />
        <span className="create-loading-line" />
        <div className="create-loading-shapes">{[0, 1, 2, 3].map(index => <span key={index} />)}</div>
        <span className="create-loading-line create-loading-line-short" />
        <span className="create-loading-line" />
      </div>
      <div className="create-loading-footer"><span className="create-loading-line" /></div>
    </aside>
    <div className="create-loading-canvas">
      <div className="create-loading-status" role="status">
        <div className="create-loading-symbol" aria-hidden="true"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4"><path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z" /></svg></div>
        <h2>{connectionFailed ? 'Studio connection interrupted' : opening ? 'Opening your studio' : 'Preparing your studio'}</h2>
        <p>{connectionFailed ? 'Check the connection in Setup. We’ll keep trying in the background.' : opening ? 'Loading your workflow and generation controls.' : 'Checking the connection to your creator.'}</p>
        <div className="create-loading-stage"><span className={connectionFailed ? 'create-connection-dot' : 'create-spinner'} aria-hidden="true" />
          <span>{connectionFailed ? 'Waiting for a connection' : backendReady ? 'ComfyUI ready · Loading studio' : 'Connecting to ComfyUI'}</span>
        </div>
        <small>Exit and Setup are available while you wait.</small>
      </div>
    </div>
    <aside className="create-loading-sidebar create-loading-effects" aria-hidden="true">
      <span className="create-loading-line create-loading-line-short" />
      {[0, 1, 2, 3].map(index => <div className="create-loading-effect" key={index}><span className="create-loading-line" /><span className="create-loading-line create-loading-line-short" /></div>)}
    </aside>
  </section>
}
