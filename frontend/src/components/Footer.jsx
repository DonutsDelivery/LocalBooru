import { Link } from 'react-router-dom'
import './Footer.css'

function Footer() {
  return (
    <footer className="site-footer">
      <div className="footer-content">
        <div className="footer-links">
          <Link to="/terms">Terms of Service</Link>
          <span className="divider">|</span>
          <Link to="/privacy">Privacy Policy</Link>
          <span className="divider">|</span>
          <a href="https://ko-fi.com/donutsdelivery" target="_blank" rel="noopener noreferrer">Support</a>
        </div>
        <div className="footer-copy">
          DonutBooru - AI Art Gallery
        </div>
      </div>
    </footer>
  )
}

export default Footer
