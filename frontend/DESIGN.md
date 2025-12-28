# DonutBooru Frontend Design Specification

## Layout Requirements

### Overall Structure
- **Fullwidth website** - maximizes screen space for images
- **Masonry style layout** - Pinterest-style grid that fills gaps efficiently
- **Support for media types**: images (jpg, png, webp), videos (mp4, webm), animated images (gif, apng)

### Header Bar
- Fixed at top of page
- Contains:
  - Logo (DonutBooru)
  - Navigation links (Browse, Hall of Fame, Rating dropdown)
  - Login with Discord button (when logged out)
  - User menu dropdown (when logged in):
    - Profile
    - My Uploads
    - Favorites
    - Moderation (if mod/admin)
    - Logout

### Left Sidebar
- **Width**: 280px
- **Behavior**:
  - Always visible when browsing search results
  - Collapses/hides when viewing fullscreen image
  - Pops out on hover when collapsed (to utilize screen space for images)
- **Contents**:
  - Tag search/filter input
  - Active filters section
  - Tags grouped by category (artist, character, copyright, general, meta)
  - Each tag shows name and count like: `female (52385)`
  - Tags are clickable to add to search
  - Tags use small font size to fit more

### Image Grid (Masonry)
- Responsive columns: 6 default, 5 at 2400px, 4 at 1800px, 3 at 1200px, 2 at 800px, 1 at 500px
- Infinite scroll loading
- Hover effects on images
- Video files show play indicator icon
- Rating badge overlay on each image

### Fullscreen/Lightbox View
- Click image to open fullscreen view
- Keyboard navigation: Arrow keys (left/right), Escape (close), i (toggle info panel)
- Navigation buttons on sides
- Counter showing current position (e.g., "5 / 100")
- Info panel slides in from left containing:
  - Image ID and rating
  - Dimensions
  - Uploader name
  - Upload date
  - Tags (clickable to search)
  - Download button

## Color Scheme (Dark Theme)
- Background primary: #1a1a2e
- Background secondary: #16213e
- Background tertiary: #0f3460
- Text primary: #e8e8e8
- Text secondary: #a0a0a0
- Accent: #e94560 (pinkish red)
- Border: #2a2a4a

### Rating Colors
- Safe: #4caf50 (green)
- Questionable: #ff9800 (orange)
- Explicit: #f44336 (red)

### Tag Category Colors
- General: #0096ff (blue)
- Artist: #ff5722 (orange-red)
- Character: #4caf50 (green)
- Copyright: #9c27b0 (purple)
- Meta: #607d8b (gray-blue)

## Tech Stack
- React 18 with Vite
- react-masonry-css for masonry layout
- react-router-dom for routing
- axios for API calls

## API Integration
- Backend API at /api (or VITE_API_URL env var)
- Endpoints used:
  - GET /images - list images with filtering
  - GET /images/:id - get image details
  - GET /tags - list tags
  - GET /auth/me - get current user
  - POST /auth/logout - logout
  - GET /auth/discord - Discord OAuth redirect

## File Structure
```
frontend/
├── src/
│   ├── components/
│   │   ├── Header.jsx/.css
│   │   ├── Sidebar.jsx/.css
│   │   ├── SearchBar.jsx/.css
│   │   ├── MasonryGrid.jsx/.css
│   │   ├── MediaItem.jsx/.css
│   │   └── Lightbox.jsx/.css
│   ├── api.js
│   ├── App.jsx/.css
│   ├── index.css
│   └── main.jsx
├── .env
├── index.html
├── package.json
└── vite.config.js
```

## Future Enhancements
- User uploads page
- Favorites/collections
- Image editing (tags)
- Moderation queue view
- Advanced search syntax
- Tag autocomplete
- Image comments
