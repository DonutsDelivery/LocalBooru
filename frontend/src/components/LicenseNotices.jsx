import license from '../../../LICENSE?raw'
import notice from '../../../NOTICE?raw'
import historicalLicense from '../../../LICENSES/MIT-legacy.txt?raw'
import trademarks from '../../../TRADEMARKS.md?raw'

const noticeStyle = { whiteSpace: 'pre-wrap', maxHeight: '20rem', overflow: 'auto' }

export default function LicenseNotices() {
  return (
    <section className="settings-section">
      <h3>License</h3>
      <p>DonutMediaCenter is free software under GNU GPL version 3 only.</p>
      <p>You may sell and redistribute copies. Covered distributions must provide corresponding source under GPLv3. This software comes without warranty.</p>
      <p><a href="https://github.com/DonutsDelivery/LocalBooru/releases" target="_blank" rel="noreferrer">Official releases and source code</a></p>
      {[
        ['Copyright and component notices', notice],
        ['GNU GPL version 3', license],
        ['Historical MIT notice', historicalLicense],
        ['Name and logo policy', trademarks],
      ].map(([title, text]) => (
        <details key={title}>
          <summary>{title}</summary>
          <pre style={noticeStyle}>{text}</pre>
        </details>
      ))}
    </section>
  )
}
